"""Transformer sul dataset definitivo: due input (A, C), due letture, tutto salvato.

Un solo script per BERT (confronto confidenziale) e ModernBERT (modello finale):
    python 02_transformer.py --modello bert-base-uncased          --max-token 256
    python 02_transformer.py --modello answerdotai/ModernBERT-base --max-token 1024

Input (addestramenti):
  A  — solo testo degli articoli
  C  — testo dell'articolo + testo delle pagine del sito della testata (in coda)
Fold: per gruppo (principale) e per testata (confronto). Impostazioni vicine al paper:
5 epoche, batch 32 (eval 16), learning rate 2e-5, AdamW.

Salva TUTTO:
  Results/2-classificazione/<nome>/<config>_<schema>_fold<k>.csv   p_l1..p_l5, y_pred, y_true, domain
  Results/2-classificazione/<nome>_<config>_<schema>_articoli.csv  (unione dei fold)
  Results/2-classificazione/<nome>_loss.csv                        loss per epoca, per ogni fold
Log: Results/2-classificazione/avanzamento_<nome>.log
Ogni fold finito viene saltato se lo script è rilanciato.
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
VERS = QUI.parent / "Dataset" / "versioni"
RESULTS = QUI.parent / "Results" / "2-classificazione"
SEED = 42
CLASSI = ["p_l1", "p_l2", "p_l3", "p_l4", "p_l5"]
SCHEMI = ["fold_gruppo", "fold_testata"]


def fissa_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


PAROLE_ART = 400   # ~512 token di articolo
PAROLE_PAG = 400   # ~512 token di pagine; articolo identico in A e C, C aggiunge le pagine


def taglia(testo, n):
    return " ".join(str(testo).split()[:n])


def carica():
    art = pd.read_csv(VERS / "dataset_finale.csv")
    art["testo"] = art["testo"].fillna("")
    fold = pd.read_csv(VERS / "fold_finale.csv")[["domain", "fold_gruppo", "fold_testata"]]
    art = art.merge(fold, on="domain", how="left")
    pag = pd.read_csv(VERS / "pagine_finale.csv")[["domain", "testo_pagine"]]
    tp = {d: taglia(t, PAROLE_PAG) for d, t in zip(pag.domain, pag.testo_pagine.fillna(""))}
    art["art_cut"] = art.testo.map(lambda t: taglia(t, PAROLE_ART))
    art["A"] = art.art_cut                                          # solo articolo (stesso troncamento)
    art["C"] = art.art_cut + "\n\n" + art.domain.map(tp).fillna("")  # articolo + pagine
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modello", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max-token", type=int, default=1024)
    ap.add_argument("--config", nargs="+", default=["A", "C"])
    ap.add_argument("--schemi", nargs="+", default=SCHEMI)
    ap.add_argument("--epoche", type=int, default=5)
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="quali fold (default tutti)")
    ap.add_argument("--early-stopping", action="store_true",
                    help="tiene da parte il 15%% dei gruppi di training come validation e ferma all'epoca migliore")
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--prova", action="store_true")
    args = ap.parse_args()

    nome = "modernbert" if "ModernBERT" in args.modello else ("bert" if "bert" in args.modello else args.modello.split("/")[-1])
    if args.early_stopping:
        nome += "_es"
    per_fold = RESULTS / nome
    per_fold.mkdir(parents=True, exist_ok=True)
    log = RESULTS / f"avanzamento_{nome}.log"
    loss_file = RESULTS / f"{nome}_loss.csv"

    def registra(m):
        riga = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {m}"
        print(riga, flush=True)
        with open(log, "a") as f:
            f.write(riga + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registra(f"=== {args.modello} (max {args.max_token} token) su {device} ===")
    art = carica()
    tok = AutoTokenizer.from_pretrained(args.modello)
    righe_loss = []
    config_corrente = None

    def tensori(sotto, con_y=True):
        enc = tok(sotto[config_corrente].tolist(), truncation=True, padding="max_length",
                  max_length=args.max_token, return_tensors="pt")
        if con_y:
            return TensorDataset(enc.input_ids, enc.attention_mask, torch.tensor(sotto.label.tolist()))
        return TensorDataset(enc.input_ids, enc.attention_mask)

    def val_loss(model, dl_val):
        model.eval(); tot = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast():
            for ids, mask, y in dl_val:
                tot += model(ids.to(device), attention_mask=mask.to(device), labels=y.to(device)).loss.item()
        return tot / len(dl_val)

    def un_fold(config, schema, fold):
        nonlocal config_corrente
        config_corrente = config
        fissa_seed(SEED)
        tr_all, te = art[art[schema] != fold], art[art[schema] == fold]
        # early stopping: tengo da parte il 15% dei GRUPPI di training come validation
        dl_val = None
        if args.early_stopping:
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
            i_tr, i_val = next(gss.split(tr_all, tr_all.label, tr_all.gruppo))
            tr, val = tr_all.iloc[i_tr], tr_all.iloc[i_val]
            dl_val = DataLoader(tensori(val), batch_size=16)
        else:
            tr = tr_all
        registra(f"{config} {schema} fold {fold}: inizio ({len(tr)} training"
                 + (f", {len(val)} validation" if args.early_stopping else "") + f", {len(te)} test)")
        dl_tr = DataLoader(tensori(tr), batch_size=32, shuffle=True)
        dl_te = DataLoader(tensori(te, con_y=False), batch_size=16)
        kw = {"reference_compile": False} if "ModernBERT" in args.modello else {}   # niente torch.compile: tempi stabili
        model = AutoModelForSequenceClassification.from_pretrained(args.modello, num_labels=5, **kw).to(device)
        opt = AdamW(model.parameters(), lr=2e-5)
        scaler = torch.cuda.amp.GradScaler()                 # precisione mista: addestramento più veloce
        miglior_val, miglior_stato, attese = float("inf"), None, 0
        for epoca in range(1, args.epoche + 1):
            model.train(); t0, tot = time.time(), 0.0
            for ids, mask, y in dl_tr:
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = model(ids.to(device), attention_mask=mask.to(device), labels=y.to(device)).loss
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
            media = tot / len(dl_tr)
            vl = val_loss(model, dl_val) if args.early_stopping else None
            righe_loss.append({"config": config, "schema": schema, "fold": fold, "epoca": epoca,
                               "loss": media, "val_loss": vl})
            registra(f"{config} {schema} fold {fold}: epoca {epoca}/{args.epoche} in {time.time()-t0:.0f} s, "
                     f"loss {media:.4f}" + (f", val_loss {vl:.4f}" if args.early_stopping else ""))
            if args.early_stopping:
                if vl < miglior_val:
                    miglior_val, miglior_stato, attese = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    attese += 1
                    if attese > args.patience:
                        registra(f"{config} {schema} fold {fold}: early stop all'epoca {epoca} (migliore: val_loss {miglior_val:.4f})")
                        break
        if args.early_stopping and miglior_stato is not None:
            model.load_state_dict(miglior_stato)
        model.eval(); probs = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for ids, mask in dl_te:
                logit = model(ids.to(device), attention_mask=mask.to(device)).logits
                probs.append(torch.softmax(logit.float(), dim=1).cpu().numpy())
        P = np.vstack(probs)
        out = pd.DataFrame(P, columns=CLASSI)
        out.insert(0, "domain", te.domain.values); out.insert(0, "article_id", te.article_id.values)
        out["y_true"] = te.label.values; out["y_pred"] = P.argmax(1)
        del model, opt; gc.collect(); torch.cuda.empty_cache()
        registra(f"{config} {schema} fold {fold}: fatto, acc articoli {(out.y_true==out.y_pred).mean():.3f}")
        return out

    if args.prova:
        sub = art.sample(300, random_state=SEED); globals()["art"] = sub  # non usato, prova veloce
        registra("PROVA: salto (usa un fold intero per un test reale)")
        return

    for config in args.config:
        for schema in args.schemi:
            parti = []
            folds = args.folds if args.folds else sorted(art[schema].unique())
            for fold in folds:
                ff = per_fold / f"{config}_{schema}_fold{fold}.csv"
                if ff.exists():
                    registra(f"{config} {schema} fold {fold}: già fatto, salto")
                else:
                    un_fold(config, schema, fold).to_csv(ff, index=False)
                parti.append(pd.read_csv(ff))
            pd.concat(parti).to_csv(RESULTS / f"{nome}_{config}_{schema}_articoli.csv", index=False)
            registra(f"=== {config} {schema}: tutti i fold pronti ===")
    if righe_loss:
        nuove = pd.DataFrame(righe_loss)
        if loss_file.exists():
            nuove = pd.concat([pd.read_csv(loss_file), nuove])
        nuove.to_csv(loss_file, index=False)
    registra("=== FINITO ===")


if __name__ == "__main__":
    main()
