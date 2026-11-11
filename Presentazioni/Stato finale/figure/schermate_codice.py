"""Schermate del codice del tentativo 20 per la presentazione (tema scuro, come quelle fatte a mano da John).
Il codice si prende dai file veri, per nome della funzione, così è sempre quello eseguito.
Uso, da questa cartella:  python schermate_codice.py"""
import ast
from pathlib import Path

import pandas as pd
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer, TextLexer

QUI = Path(__file__).resolve().parent
T20 = QUI.parents[2] / "Code/Pipeline"
D20 = QUI.parents[2] / "Dataset/tentativi/20_FromScratch"
OUT = QUI / "codice"; OUT.mkdir(exist_ok=True)


def funzione(file, nome, da=None, a=None):
    """Il sorgente della funzione (o della costante) `nome` in `file`; con `da`/`a` solo le righe tra le due che
    contengono quei testi (comprese)."""
    testo = (T20 / file).read_text()
    for nodo in ast.parse(testo).body:
        n = getattr(nodo, "name", None) or (nodo.targets[0].id if isinstance(nodo, ast.Assign) else None)
        if n == nome:
            righe = testo.splitlines()[nodo.lineno - 1:nodo.end_lineno]
            break
    else:
        raise KeyError(nome)
    if da:
        righe = righe[next(i for i, r in enumerate(righe) if da in r):]
    if a:
        righe = righe[:next(i for i, r in enumerate(righe) if a in r) + 1]
    rientro = min(len(r) - len(r.lstrip()) for r in righe if r.strip())
    return "\n".join(r[rientro:] for r in righe)


def immagine(codice, nome, lexer=PythonLexer()):
    fmt = ImageFormatter(style="one-dark", font_name="DejaVu Sans Mono", font_size=22, line_numbers=False, image_pad=18,
                         line_pad=6)
    (OUT / f"{nome}.png").write_bytes(highlight(codice, lexer, fmt))


# ------------------------------------------------------------------------------------------------ i modelli
immagine(funzione("models.py", "_matrices", da="if name"), "tfidf_matrici")
immagine(funzione("models.py", "_fit", da="from sklearn"), "modelli_lineari")
immagine(funzione("models.py", "qwen_embeddings", da="texts = [f", a="E[idx] = h"), "qwen")
immagine(funzione("models.py", "mistral_embeddings", da="quant = BitsAndBytesConfig", a="device_map")
         + "\n...\n" + funzione("models.py", "mistral_embeddings", da="out = m.model", a="E[l][idx] ="), "mistral")
immagine(funzione("models.py", "bert_fold", da="model = AutoModelForSequenceClassification", a="label_smoothing")
         + "\n...\n" + funzione("models.py", "bert_fold", da="z = model(input_ids", a="** 2).mean()"), "modernbert")
immagine("".join(f"{r.model:12s} mediana {r['median']:5d}   99% {r['99%']:5d}   max {r['max']:5d}   limite {r.limit:5d}   troncati {r.truncated}\n"
                 for _, r in pd.read_csv(D20 / "token_lengths.csv").iterrows()), "token", TextLexer())

# ------------------------------------------------------------------------------------------------ l'addestramento
immagine(funzione("models.py", "score_split"), "split")
immagine(funzione("models.py", "GRIDS"), "griglie")
immagine(funzione("models.py", "linear_fold", da="# 1. three inner folds", a="m = _fit(task, best"), "fold_lineare")
immagine(funzione("models.py", "bert_fold", da="accumulate = max", a="sched.step()"), "addestramento_bert")
immagine(funzione("models.py", "reproducible", da="import random"), "riproducibile")

# ------------------------------------------------------------------------------------------------ la valutazione
immagine(funzione("metrics.py", "outlets_classifier", da="g = pa.groupby") + "\n\n"
         + funzione("metrics.py", "outlets_regressor", da="T = pd.DataFrame"), "da_articoli_a_testata")
immagine(funzione("metrics.py", "levels_quotas", da="predicted, fold, true_level ="), "quote")
immagine(funzione("metrics.py", "expected_score", da="out = pd.Series"), "voto_atteso")
immagine(funzione("metrics.py", "bootstrap", da="y, p = np.asarray"), "bootstrap")

# ------------------------------------------------------------------------------------------------ solo regressori (5 novembre)
fit = funzione("models.py", "_fit").splitlines()
immagine("\n".join([fit[0], "    ...", next(r for r in fit if "return Ridge" in r)]), "ridge")
immagine(funzione("models.py", "bert_fold", da="model = AutoModelForSequenceClassification", a="reference_compile")
         + "\n...\n" + funzione("models.py", "bert_fold", da="loss = ((z[:, 0]", a="** 2).mean()"), "modernbert_reg")
immagine("\n".join(r for r in funzione("models.py", "GRIDS").splitlines() if "classification" not in r), "griglie_reg")
immagine(funzione("metrics.py", "outlets_regressor"), "articoli_testata_reg")
immagine(funzione("metrics.py", "remapped_score"), "stirare")
immagine(funzione("models.py", "bert_fold", da="reproducible(SEED + k)", a="reproducible(SEED + k)") + "\n...\n"
         + funzione("models.py", "bert_fold", da="def batches", a="return groups"), "seme_bert")
immagine(funzione("mbfc.py", "quotas") + "\n\n" + funzione("mbfc.py", "medians"), "mbfc_quote_mediane")
immagine(funzione("mbfc.py", "thresholds"), "mbfc_soglie")
immagine(funzione("mbfc.py", "by_fold"), "mbfc_fold")
immagine(funzione("models.py", "bert_fold", da="accumulate = max", a="steps = epochs") + "\n...\n"
         + funzione("models.py", "bert_fold", da="scaler.scale(loss", a="sched.step()"), "addestramento_bert_reg")
print(sorted(p.name for p in OUT.glob("*.png")))
