"""Step 3, part 2 (notebook 03b; checks in 03c): a stronger anonymization, on top of the six rules of part 1
(`anonymization.py`), so that the claim "the models do not see who published the article" is hard to attack. The text is never rewritten by a model: every change is a
deletion of a repeated boilerplate sentence or a replacement of an exact phrase with a placeholder, so every change can
be listed and checked.

1. Boilerplate: a sentence (normalized: lowercase, digits as 0) that appears in at least `BOILER_SHARE` of the articles
   of an outlet (and in at least `BOILER_MIN` of them) is removed from that outlet's articles (newsletter and donation
   calls, disclaimers, fixed sign-offs, "follow us on ..."). Sentences shorter than `BOILER_WORDS` words are kept.
2. LLM detection (`DETECTOR`, an instruction model run locally, greedy decoding): for every article, given its outlet's
   domain, the model lists in JSON the phrases that reveal the outlet: its name in any form (short, acronym, possessive,
   branded shows, newsletters, podcasts) and its own staff. Kept only if the phrase is really in the article and it is not
   a common word or a public name: a phrase that appears (lowercase, as a word) in more than `COMMON_SHARE` of the
   articles of OTHER outlets is dropped (e.g. "Trump", "News"). The phrases kept form a lexicon per outlet, applied to
   ALL its articles: outlet names -> [OUTLET] (any case), staff -> [AUTHOR] (exact case).
3. Checks (notebook 03c), on a fixed sample (`SAMPLE` articles, seeded), on the original text, the step 3 text and the new text:
   - an LLM attacker (`CHECKER`, the larger model) asked which website published the article (accuracy = right domain);
   - a classifier that tries to recognize the outlet among the 226 (TF-IDF + linear SVM, 80/20 of the
     articles of every outlet);
   - the agreement of the detector with a larger model (`CHECKER`) on `CHECK_SAMPLE` articles;
   - a sheet of `AUDIT` random articles with every change highlighted, for a check by hand.
Output: Dataset/tentativi/20_FromScratch/articles_anonymous.csv.gz (same ids and labels as step 2), the text of every
later step; work files in anonymization/."""
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd

from common import OUT, SEED, log

WORK = OUT / "anonymization"                     # the files of part 2 and of the checks (answers, lexicon, audit)
LOG = WORK / "log.txt"
DETECTOR = "Qwen/Qwen2.5-7B-Instruct"
CHECKER = "mistralai/Mistral-Small-24B-Instruct-2501"
BOILER_SHARE, BOILER_MIN, BOILER_WORDS = 0.15, 3, 3
COMMON_SHARE = 0.01
SAMPLE, CHECK_SAMPLE, AUDIT = 678, 400, 100           # 678 = 3 articles per outlet
PLACEHOLDER = re.compile(r"\[(?:OUTLET|AUTHOR|EMAIL|URL|ACCOUNT)\]")
GENERIC = {"the", "news", "daily", "times", "post", "media", "staff", "editor", "editors", "editorial", "team", "we",
           "our", "us", "reporter", "reporters", "writer", "contributor", "newsletter", "podcast", "show", "website",
           "site", "blog", "magazine", "journal", "network", "press", "today", "report", "outlet"}

PROMPT = """The news article below was published by the outlet whose website is {domain}.
List every phrase in the TITLE or TEXT that reveals which outlet published it:
- "outlet": the outlet's own name in any form (full, shortened, acronym, possessive), and names of its own shows,
  newsletters, podcasts, sections or brands;
- "staff": names of the outlet's own journalists, editors or staff presented as such (bylines, "reporter X",
  "X contributed to this report", "X for <outlet>").
Do NOT list other people, other organizations, places or ordinary words. Copy every phrase exactly as written.
Answer only with JSON: {{"outlet": [...], "staff": [...]}}; if there is nothing, {{"outlet": [], "staff": []}}.

TITLE: {title}

TEXT:
{text}"""

ATTACK = """Which news website most likely published the article below? Answer with its domain only (for example
example.com), or "unknown" if you cannot tell.

TITLE: {title}

TEXT:
{text}"""


# ------------------------------------------------------------------------- 1. boilerplate
SENT = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"“‘'\[(])")


def _key(s):
    return re.sub(r"\d", "0", re.sub(r"\s+", " ", s.lower())).strip()


def boilerplate(art):
    """Remove from every outlet the sentences repeated in >= BOILER_SHARE of its articles. Returns the new texts and
    the table of the removed sentences (outlet, sentence, articles)."""
    split = [[SENT.split(p) for p in str(t).split("\n")] for t in art.text.fillna("")]
    removed, new = [], [None] * len(art)
    for d, idx in art.groupby("domain").indices.items():
        counts = Counter()
        for i in idx:
            counts.update({_key(s) for par in split[i] for s in par if len(s.split()) >= BOILER_WORDS})
        limit = max(BOILER_MIN, BOILER_SHARE * len(idx))
        bad = {k for k, c in counts.items() if c >= limit}
        removed += [{"domain": d, "sentence": k, "articles": counts[k], "of": len(idx)} for k in bad]
        for i in idx:
            pars = [" ".join(s for s in par if _key(s) not in bad) for par in split[i]]
            new[i] = "\n".join(p for p in pars if p.strip())
    return pd.Series(new, index=art.index), pd.DataFrame(removed)


# ------------------------------------------------------------------------- 2. LLM detection
def load_llm(name, four_bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_compute_dtype=torch.float16)} if four_bit else {}
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map={"": 0}, **kw).eval()
    return tok, m


def generate(tok, m, prompts, max_new_tokens, budget=24000, max_batch=32, done=None, save=None, logfile=None):
    """Greedy answers to `prompts` (chat format), in batches of similar length. `done`: {index: answer} already
    computed (resumes); `save(done)` is called every 20 batches."""
    import torch
    done = dict(done or {})
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    lengths = np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])
    order = [i for i in np.argsort(lengths) if i not in done]
    batches, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * (lengths[i] + max_new_tokens) > budget or len(cur) >= max_batch):
            batches.append(cur); cur = []
        cur.append(i)
    batches += [cur] if cur else []
    start = time.time()
    for b, idx in enumerate(batches, 1):
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, o in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            done[int(i)] = tok.decode(o, skip_special_tokens=True)
        if save and (b % 20 == 0 or b == len(batches)):
            save(done)
            log(f"LLM: {len(done):,}/{len(prompts):,} ({time.time() - start:.0f} s)", logfile)
    return done


def parse(answer):
    """The two lists of the JSON answer (tolerant: the first {...} block; on failure, empty lists)."""
    m = re.search(r"\{.*\}", answer, flags=re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    get = lambda k: [str(x).strip() for x in d.get(k, []) if isinstance(x, (str, int)) and str(x).strip()]
    return get("outlet"), get("staff")


def detection_prompts(art, max_words=1500):
    return [PROMPT.format(domain=d, title=t if isinstance(t, str) else "", text=" ".join(str(x).split()[:max_words]))
            for d, t, x in zip(art.domain, art.title, art.text)]


# ------------------------------------------------------------------------- lexicon and replacement
def _word_re(phrase, ignore_case):
    return re.compile(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", flags=re.I if ignore_case else 0)


def lexicon(art, answers):
    """Per outlet: the outlet phrases and the staff phrases kept (see the module docstring)."""
    rows = []
    for i, a in answers.items():
        outlet, staff = parse(a)
        text = f"{art.title.iloc[i]}\n{art.text.iloc[i]}"
        for kind, phrases in [("outlet", outlet), ("staff", staff)]:
            for p in phrases:
                p = PLACEHOLDER.sub("", p).strip(" .,;:'\"“”’()")
                if len(p) < 2 or p.lower() in GENERIC or all(w.lower() in GENERIC for w in p.split()) or p not in text:
                    continue
                rows.append({"domain": art.domain.iloc[i], "kind": kind, "phrase": p})
    L = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["domain", "kind", "phrase"])
    # common words and public names: frequent (lowercase, as a word) in the articles of OTHER outlets
    low = art.text.fillna("").str.lower().to_numpy()
    dom = art.domain.to_numpy()
    keep = []
    for (d, p), _ in L.groupby(["domain", "phrase"]):
        q = p.lower(); r = _word_re(q, False)
        others = dom != d
        n = sum(1 for t in low[others] if q in t and r.search(t))       # fast substring test first
        keep.append({"domain": d, "phrase": p, "elsewhere": n, "kept": n <= COMMON_SHARE * others.sum()})
    K = pd.DataFrame(keep, columns=["domain", "phrase", "elsewhere", "kept"])
    return L.merge(K, on=["domain", "phrase"])


def apply_lexicon(art, L):
    """Replace, in every article of an outlet, its kept phrases (longest first). Returns texts, titles, counts."""
    texts, titles = art.text.fillna("").copy(), art.title.fillna("").copy()
    counts = pd.Series(0, index=art.index)
    for d, g in L[L.kept].groupby("domain"):
        idx = art.index[art.domain == d]
        for _, r in g.assign(n=g.phrase.str.len()).sort_values("n", ascending=False).iterrows():
            pat = _word_re(r.phrase, ignore_case=r.kind == "outlet")
            ph = "[OUTLET]" if r.kind == "outlet" else "[AUTHOR]"
            for col in (texts, titles):
                new = col[idx].map(lambda t: pat.subn(ph, t))
                counts[idx] += new.map(lambda x: x[1])
                col[idx] = new.map(lambda x: x[0])
    return texts, titles, counts


# ------------------------------------------------------------------------- 3. checks
def sample(art, per_outlet=3, seed=SEED):
    return art.groupby("domain", group_keys=False).sample(n=per_outlet, random_state=seed).index


def attack_accuracy(answers, domains):
    """Share of articles whose outlet the attacker names (registered domain, without www)."""
    def norm(s):
        s = re.sub(r"^https?://", "", str(s).strip().lower().strip("\"'`.,;:()[]<> "))
        return re.sub(r"^www\.", "", s.split("/")[0])
    hits = [norm(a.split()[0] if a.split() else "") == norm(d) for a, d in zip(answers, domains)]
    return float(np.mean(hits))


def outlet_classifier(texts, domains, seed=SEED):
    """Accuracy of recognizing the outlet among all (TF-IDF + linear SVM), 80/20 of each outlet's articles."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(texts)), test_size=0.2, stratify=domains, random_state=seed)
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    A, B = v.fit_transform(np.asarray(texts)[tr]), v.transform(np.asarray(texts)[te])
    m = LinearSVC(C=0.5, random_state=seed).fit(A, np.asarray(domains)[tr])
    return float((m.predict(B) == np.asarray(domains)[te]).mean())


def audit_sheet(before, after, ids, domains, path):
    """HTML with the step 3 text and the new text side by side; placeholders highlighted."""
    import html
    mark = lambda t: re.sub(r"(\[(?:OUTLET|AUTHOR|EMAIL|URL|ACCOUNT)\])", r"<b style='background:#ffe08a'>\1</b>", html.escape(t))
    parts = ["<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:1400px'>",
             "<h2>Audit of the anonymization: for every article, write in the CSV the identifiers left and the wrong replacements</h2>"]
    for i, d, a, b in zip(ids, domains, before, after):
        parts.append(f"<h3>article {i} — outlet {html.escape(d)}</h3><table><tr><td style='width:50%;vertical-align:top'>"
                     f"<i>step 3</i><br>{mark(a)}</td><td style='vertical-align:top'><i>step 3b</i><br>{mark(b)}</td></tr></table><hr>")
    path.write_text("".join(parts) + "</body></html>")
