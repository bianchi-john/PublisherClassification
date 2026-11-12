"""Step 3: anonymization, in three passes on every title and text:
1. links and e-mail addresses -> [URL], [EMAIL] (regular expressions of `anonymization.py`);
2. the outlet's own name, written from its domain, joined or spaced, with or without "the" -> [OUTLET]
   (`anonymization.py`; a name that is also a common word is replaced only when capitalized);
3. a standard named-entity recognizer, GLiNER (zero-shot NER; Zaratiana et al., NAACL 2024), with the labels
   "news outlet" and "journalist": every span with score >= `THRESHOLD` -> [OUTLET] / [AUTHOR] (every outlet and every
   journalist, not only the publisher and its staff).
Nothing else is changed: no rewriting. GLiNER reads at most 384 tokens, so the text is read in windows of `WINDOW`
words overlapping by `OVERLAP` words; overlapping spans: the one with the higher score wins. A span is replaced only if
it does not touch a placeholder of passes 1-2 and contains a capitalized word that is not a pronoun (`keep`: GLiNER
also tags "I", "she", "reporters" and the placeholders themselves). Passes 1-2 are needed
because GLiNER alone leaves the outlet's own name in about 1 article in 4 (e.g. "AmmoLand previously covered").
Check: how many articles still contain the name of their own outlet."""
import re
import time

import numpy as np
import pandas as pd

from common import OUT, SEED, log

WORK = OUT / "anonymization"
LOG = WORK / "log.txt"
MODEL = "urchade/gliner_large-v2.1"
LABELS = {"news outlet": "[OUTLET]", "journalist": "[AUTHOR]"}
THRESHOLD = 0.5
WINDOW, OVERLAP = 200, 30


def windows(text):
    """(start, end) character offsets of overlapping windows of WINDOW words."""
    words = [m.span() for m in re.finditer(r"\S+", text)]
    if not words:
        return []
    out, i = [], 0
    while True:
        j = min(i + WINDOW, len(words))
        out.append((words[i][0], words[j - 1][1]))
        if j == len(words):
            return out
        i = j - OVERLAP


def detect(texts, model, batch=16, logfile=None):
    """GLiNER spans of every text: list (per text) of (start, end, label, score), overlaps resolved."""
    pieces = [(t, s, e) for t, text in enumerate(texts) for s, e in windows(text)]
    found = [[] for _ in texts]
    start = time.time()
    for b in range(0, len(pieces), batch):
        chunk = pieces[b:b + batch]
        res = model.batch_predict_entities([texts[t][s:e] for t, s, e in chunk], list(LABELS), threshold=THRESHOLD)
        for (t, s, _), ents in zip(chunk, res):
            found[t] += [(s + x["start"], s + x["end"], x["label"], float(x["score"])) for x in ents]
        if logfile and (b // batch) % 500 == 0:
            log(f"GLiNER: {b:,}/{len(pieces):,} windows ({time.time() - start:.0f} s)", logfile)
    out = []
    for spans in found:
        kept = []
        for sp in sorted(set(spans), key=lambda x: (-x[3], x[0])):        # higher score first
            if all(sp[1] <= k[0] or sp[0] >= k[1] for k in kept):
                kept.append(sp)
        out.append(sorted(kept))
    return out


def replace(text, spans):
    """The text with every span replaced by the placeholder of its label (from the end, offsets stay valid)."""
    for s, e, label, _ in sorted(spans, reverse=True):
        text = text[:s] + LABELS[label] + text[e:]
    return text


def run(art, logfile=None, block=2000):
    """Spans of titles and texts, computed in blocks saved to WORK (a re-run resumes)."""
    from gliner import GLiNER
    import models as MO
    WORK.mkdir(parents=True, exist_ok=True)
    model = None
    rows = []
    for b in range(0, len(art), block):
        f = WORK / f"spans_{b // block:03d}.csv.gz"
        if not f.exists():
            if model is None:
                MO.wait_for_gpu(6, logfile)
                MO.reproducible(SEED)
                model = GLiNER.from_pretrained(MODEL).to("cuda").eval()
            part = art.iloc[b:b + block]
            r = []
            for field in ["title", "text"]:
                for aid, spans in zip(part.article_id, detect(part[field].fillna("").tolist(), model, logfile=logfile)):
                    r += [{"article_id": aid, "field": field, "start": s, "end": e, "label": lab, "score": sc}
                          for s, e, lab, sc in spans]
            pd.DataFrame(r, columns=["article_id", "field", "start", "end", "label", "score"]).to_csv(f, index=False)
            log(f"GLiNER: articles {b + len(part):,}/{len(art):,}", logfile)
        rows.append(pd.read_csv(f))
    return pd.concat(rows, ignore_index=True)


def rules(art):
    """Passes 1-2: links, e-mails and the outlet's own name (from its domain). Returns the new articles and the counts."""
    import anonymization as AN
    names = {d: AN.outlet_names(d) for d in sorted(art.domain.unique())}
    common = AN.common_words(names, art.text, art.domain)
    lower = AN.lowercase_names(names, common, art.text, art.domain)
    new, counts = art.copy(), {"email": 0, "url": 0, "name": 0}
    for d, idx in art.groupby("domain").groups.items():
        name_re = AN.name_pattern(names[d], common, lower)
        for field in ["title", "text"]:
            out = []
            for t in art.loc[idx, field].fillna(""):
                for rule, pat, token in [("email", AN.EMAIL, "[EMAIL]"), ("url", AN.URL, "[URL]"), ("name", name_re, "[OUTLET]")]:
                    t, n = pat.subn(token, t); counts[rule] += n
                out.append(t)
            new.loc[idx, field] = out
    return new, counts


PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
            "their", "it", "its", "the", "a", "an"}
PLACEHOLDER = re.compile(r"\[?(?:OUTLET|AUTHOR|EMAIL|URL)\]?")


def keep(text, start, end, label):
    """GLiNER also tags placeholders already in the text, pronouns ("I", "she") and common nouns ("reporters"): a span
    is kept only if it does not touch a placeholder of passes 1-2 and contains a capitalized word that is not a
    pronoun (a name: "Megyn Kelly", "JVL", "Fox News")."""
    span = text[start:end]
    if PLACEHOLDER.search(text[max(0, start - 1):end + 1]):
        return False
    words = re.findall(r"[^\W\d_][\w'’.-]*", span)
    return any(w[0].isupper() and len(w) >= 2 and w.lower() not in PRONOUNS for w in words)


def apply(art, spans):
    """Titles and texts with the kept spans replaced (see `keep`). Adds the column `kept` to `spans`."""
    text = {(a, f): t for f in ["title", "text"] for a, t in zip(art.article_id, art[f].fillna(""))}
    spans["kept"] = [keep(text[(a, f)], s, e, lab) for a, f, s, e, lab in
                     zip(spans.article_id, spans.field, spans.start, spans.end, spans.label)]
    spans = spans[spans.kept]
    out = art.copy()
    for field in ["title", "text"]:
        g = spans[spans.field == field].groupby("article_id")
        by = {a: list(zip(d.start, d.end, d.label, d.score)) for a, d in g}
        out[field] = [replace(t, by.get(a, [])) for a, t in zip(art.article_id, art[field].fillna(""))]
    return out


def own_name_left(art):
    """Articles that still contain the name of their own outlet (from its domain; joined or spaced, any case)."""
    import anonymization as AN
    hits = []
    for d, g in art.groupby("domain"):
        pat = AN.name_pattern(AN.outlet_names(d), set())
        hits += [(d, a, bool(pat.search(f"{t}\n{x}"))) for a, t, x in zip(g.article_id, g.title.fillna(""), g.text.fillna(""))]
    return pd.DataFrame(hits, columns=["domain", "article_id", "own name left"])
