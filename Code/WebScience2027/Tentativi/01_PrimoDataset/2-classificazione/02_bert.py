"""BERT sulle versioni pulite del dataset, con gli stessi fold del classificatore a feature.

Impostazioni identiche a quelle del paper (1-classification/2 - Reliability - 5Fold - Separate.ipynb):
bert-base-uncased, 256 token, 5 epoche, batch 32 (valutazione 16), learning rate 2e-5, AdamW,
nessun peso per classe.

Ogni fold viene salvato appena finisce (Results/2-classificazione/bert/): se lo script si interrompe,
rilanciandolo riparte dal primo fold mancante. Quando tutti i fold di una combinazione sono pronti,
scrive 02_bert_<versione>_<schema>_articoli.csv, nello stesso formato del classificatore a feature.

Uso:
    python 02_bert.py                                   # tutto: fold per gruppo prima, poi per testata
    python 02_bert.py --versioni min0 --schemi fold_gruppo
    python 02_bert.py --prova                           # verifica veloce: 300 articoli, 1 epoca, 1 fold
Avanzamento:
    tail -F ../Results/2-classificazione/avanzamento_bert.log
"""
import argparse
import gc
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

QUI = Path(__file__).resolve().parent
PULITO = QUI.parents[4] / "Dataset" / "tentativi/01_PrimoDataset/DatasetPulito"
RESULTS = QUI.parent / "Results" / "2-classificazione"
PER_FOLD = RESULTS / "bert"
LOG = RESULTS / "avanzamento_bert.log"

MODELLO = "bert-base-uncased"
MAX_TOKEN = 256
BATCH_TRAIN, BATCH_EVAL = 32, 16
LEARNING_RATE = 2e-5
SEED = 42


def registra(messaggio):
    riga = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {messaggio}"
    print(riga, flush=True)
    with open(LOG, "a") as f:
        f.write(riga + "\n")


def fissa_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def carica_versione(versione):
    df = pd.read_csv(PULITO / f"dataset_{versione}.csv")
    df["testo"] = df["testo"].fillna("")
    fold = pd.read_csv(PULITO / f"fold_{versione}.csv")
    return df.merge(fold, on="domain", how="left")


def codifica(tokenizer, testi, etichette):
    enc = tokenizer(testi, truncation=True, padding="max_length", max_length=MAX_TOKEN, return_tensors="pt")
    return TensorDataset(enc.input_ids, enc.attention_mask, torch.tensor(etichette))


def un_fold(df, schema, fold, epoche, etichetta, device):
    fissa_seed(SEED)
    train = df[df[schema] != fold]
    test = df[df[schema] == fold]
    registra(f"{etichetta}: inizio ({len(train)} articoli di training, {len(test)} di test)")

    tokenizer = AutoTokenizer.from_pretrained(MODELLO)
    train_loader = DataLoader(codifica(tokenizer, train.testo.tolist(), train.label.tolist()),
                              batch_size=BATCH_TRAIN, shuffle=True)
    test_loader = DataLoader(codifica(tokenizer, test.testo.tolist(), test.label.tolist()),
                             batch_size=BATCH_EVAL)

    model = AutoModelForSequenceClassification.from_pretrained(MODELLO, num_labels=5).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoca in range(1, epoche + 1):
        model.train()
        inizio, totale = time.time(), 0.0
        for passo, (input_ids, mask, etichette) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = model(input_ids.to(device), attention_mask=mask.to(device), labels=etichette.to(device)).loss
            loss.backward()
            optimizer.step()
            totale += loss.item()
            if passo % 100 == 0:
                registra(f"{etichetta}: epoca {epoca}/{epoche}, passo {passo}/{len(train_loader)}, loss {totale / passo:.4f}")
        registra(f"{etichetta}: epoca {epoca}/{epoche} finita in {time.time() - inizio:.0f} s, loss media {totale / len(train_loader):.4f}")

    model.eval()
    pred = []
    with torch.no_grad():
        for input_ids, mask, _ in test_loader:
            pred.extend(model(input_ids.to(device), attention_mask=mask.to(device)).logits.argmax(dim=1).cpu().tolist())

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = pd.DataFrame({"article_id": test.article_id, "domain": test.domain, "gruppo": test.gruppo,
                        "fold": fold, "y_true": test.label, "y_pred": pred})
    registra(f"{etichetta}: fatto, accuratezza per articolo {(out.y_true == out.y_pred).mean():.3f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versioni", nargs="+", default=["min0", "min10", "min20"])
    parser.add_argument("--schemi", nargs="+", default=["fold_gruppo", "fold_testata"])
    parser.add_argument("--epoche", type=int, default=5)
    parser.add_argument("--prova", action="store_true", help="300 articoli, 1 epoca, solo il fold 1; non salva nulla")
    args = parser.parse_args()

    PER_FOLD.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registra(f"=== avvio su {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) ===")

    if args.prova:
        df = carica_versione("min0").sample(300, random_state=SEED)
        out = un_fold(df, "fold_gruppo", 1, 1, "PROVA", device)
        registra(f"PROVA riuscita: {len(out)} predizioni. Lo script è pronto.")
        return

    for schema in args.schemi:
        for versione in args.versioni:
            df = carica_versione(versione)
            parti = []
            for fold in sorted(df[schema].unique()):
                file_fold = PER_FOLD / f"{versione}_{schema}_fold{fold}.csv"
                etichetta = f"{versione} {schema} fold {fold}"
                if file_fold.exists():
                    registra(f"{etichetta}: già fatto, salto")
                else:
                    un_fold(df, schema, fold, args.epoche, etichetta, device).to_csv(file_fold, index=False)
                parti.append(pd.read_csv(file_fold))
            pd.concat(parti).to_csv(RESULTS / f"02_bert_{versione}_{schema}_articoli.csv", index=False)
            registra(f"=== {versione} {schema}: tutti i fold pronti ===")


if __name__ == "__main__":
    main()
