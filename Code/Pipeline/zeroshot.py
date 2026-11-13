"""Step 13: an instruction-tuned language model asked directly, without any training, to estimate the NewsGuard score
of the outlet that published an article (zero-shot). The baseline for "just ask the model": the same family as the
frozen Mistral-Small-24B Base of step 7, in its instruct version, reading the same anonymous text.

- Prompt: what NewsGuard is and its nine criteria, the placeholders of the anonymization, and the request to answer
  with one integer from 0 to 100 (`SYSTEM`, `USER`).
- Decoding: greedy (deterministic), at most 6 new tokens; the first integer of the answer, clipped to 0-100. An answer
  without an integer is kept as missing and counted.
- The score of an outlet is the mean over its articles, as for every other model. Nothing is trained, so the folds only
  organize the outputs: predictions/regression/LLM-zero-shot/fold<k>.csv.gz, like the models of step 7.
"""
import re
import time

import numpy as np
import pandas as pd

from common import OUT, PREDICTIONS, SEED, log
from models import MAX_TOKENS, batches_by_tokens, load_4bit, reproducible

MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
NAME = "LLM-zero-shot"
FOLDER = PREDICTIONS / "regression" / NAME
ANSWERS = OUT / "zero_shot_answers.csv.gz"

SYSTEM = """You are an expert media analyst. NewsGuard rates news and information websites with a score from 0 to 100, \
the weighted sum of nine journalistic criteria: the site does not repeatedly publish false content (22 points), \
gathers and presents information responsibly (18), regularly corrects or clarifies errors (12.5), handles the \
difference between news and opinion responsibly (12.5), avoids deceptive headlines (10), discloses ownership and \
financing (7.5), clearly labels advertising (7.5), reveals who is in charge, including possible conflicts of interest \
(5), and provides the names of content creators with contact or biographical information (5). Scores below 60 mean \
that the site generally fails to meet basic standards of credibility and transparency."""

USER = """Below is one article published by a news website. The names of news outlets have been replaced by [OUTLET], \
the names of journalists by [AUTHOR], links by [URL] and e-mail addresses by [EMAIL].

Estimate the NewsGuard score (0-100) of the website that published this article. Answer with a single integer and \
nothing else.

Article:
{text}"""


def parse(answer):
    """The first integer of the answer, clipped to 0-100; NaN if there is none."""
    m = re.search(r"\d+", answer or "")
    return float(min(100, max(0, int(m.group())))) if m else np.nan


def exact(answer):
    """True if the answer is just an integer (the parsing above did not have to guess)."""
    return bool(re.fullmatch(r"\s*\d+\s*", answer or ""))


def answers(texts, logfile, budget=24576, max_size=8, partial=None, save_every=50):
    """The raw answer of the model for every text (resumes from `partial`, a .csv.gz of done answers)."""
    import torch

    reproducible(SEED)
    tok, m = load_4bit(MODEL, "left")
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": USER.format(text=t)}],
                                       tokenize=False, add_generation_prompt=True) for t in texts]
    lengths = np.array([len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]])
    assert lengths.max() < MAX_TOKENS + 1024, "a prompt is longer than expected"
    out = np.full(len(texts), None, dtype=object)
    if partial is not None and partial.exists():
        p = pd.read_csv(partial, keep_default_na=False)
        out[p.row.to_numpy()] = p.answer.to_numpy(); log(f"zero-shot: resuming, {len(p):,} answers done", logfile)
    start = time.time()
    with torch.no_grad():
        for b, idx in enumerate(batches_by_tokens(lengths, budget, max_size)):
            if all(out[i] is not None for i in idx):
                continue
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = m.generate(**enc, max_new_tokens=6, do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id)
            for i, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                out[i] = tok.decode(g, skip_special_tokens=True).strip()
            if b % save_every == 0:
                done = np.array([o is not None for o in out])
                log(f"zero-shot: {done.sum():,}/{len(texts):,} articles ({time.time() - start:.0f} s)", logfile)
                if partial is not None:
                    pd.DataFrame({"row": np.where(done)[0], "answer": out[done]}).to_csv(partial, index=False)
    del m; torch.cuda.empty_cache()
    return out


def run(art, logfile, limit=None):
    """Answers for every article (or the first `limit`, for a quick test), saved like the predictions of step 7; read
    back if already done."""
    if limit is None and ANSWERS.exists() and all((FOLDER / f"fold{k}.csv.gz").exists() for k in range(1, 6)):
        log("zero-shot: already done", logfile)
        return pd.read_csv(ANSWERS, keep_default_na=False, na_values={"prediction": [""]})
    a = art if limit is None else art.iloc[:limit]
    raw = answers(a.text.fillna("").tolist(), logfile, partial=None if limit else OUT / "zero_shot_partial.csv.gz")
    res = a[["article_id", "domain", "fold"]].assign(answer=raw, prediction=[parse(x) for x in raw])
    if limit:
        return res
    res.to_csv(ANSWERS, index=False)
    FOLDER.mkdir(parents=True, exist_ok=True)
    for k in range(1, 6):
        res[res.fold == k][["article_id", "domain", "fold", "prediction"]].to_csv(FOLDER / f"fold{k}.csv.gz", index=False)
    (OUT / "zero_shot_partial.csv.gz").unlink(missing_ok=True)
    log(f"zero-shot: done, {res.prediction.isna().sum()} answers without an integer, "
        f"{(~res.answer.map(exact)).sum()} not just an integer", logfile)
    return res
