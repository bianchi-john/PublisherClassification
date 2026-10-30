"""Step 3: anonymization. Remove from text and title everything that says WHO published the article (its name, its
authors, its links), so that a model cannot recognise the outlet and use what it already knows about it. Nothing that
says WHAT the article is about is touched: people, places, facts and other outlets stay.

Six simple rules, in this order:
1. outlet signature (the title tail repeated in >= 30% of its titles: "... | The New York Sun")  -> [OUTLET]
2. e-mail addresses                                                                              -> [EMAIL]
3. links (http..., www..., bare domains such as bellingcat.com)                                  -> [URL]
4. social accounts (@name)                                                                       -> [ACCOUNT]
5. the outlet's own authors (from the author field; one-word names only after "by")              -> [AUTHOR]
6. the outlet's name, from its domain (joined or spaced, with or without "the")                  -> [OUTLET]
A name that is also a common word (politics, time, reason) is removed only when capitalised, unless the outlet writes
its own name in lowercase in >= 80% of its articles (only spiked).
"""
import collections
import re

import numpy as np
import pandas as pd

from common import SEED

PLATFORMS = {"substack"}          # hosted sites: the outlet name is the subdomain (chriscillizza.substack.com)
COMMON_WORD_COUNT = 20            # a name is a common word if it appears lowercase > 20 times in OTHER outlets
LOWERCASE_SHARE = 0.8             # ... unless the outlet writes it lowercase in >= 80% of its own articles
SIGNATURE_SHARE = 0.3             # a title tail repeated in >= 30% of the titles (and >= 3) is the signature
NOT_A_PERSON = {"press", "news", "staff", "editor", "editors", "editorial", "team", "board", "reuters", "agency",
                "media", "service", "services", "desk", "wire", "newsroom", "report", "reports", "contributor",
                "contributors", "guest", "admin", "administrator", "writer", "writers", "the"}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?i:https?://|www\.)\S+"                                   # full links
                 r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|info|news|media|press|co|uk|us|ca|au|nz|cn|eu|io|tv|"
                 r"scot|su|one)\b(?:/\S*)?")                                 # bare domains, lowercase suffix
ACCOUNT = re.compile(r"(?<![\w@])@\w{2,}")
WORD_EDGE_L, WORD_EDGE_R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"


def _without_links(texts):
    return pd.Series(texts).astype(str).map(lambda t: ACCOUNT.sub(" ", URL.sub(" ", EMAIL.sub(" ", t))))


# ------------------------------------------------------------------------- 1. the outlet name
def outlet_names(domain):
    """Names from the domain: the part before the suffix (100percentfedup.com -> 100percentfedup, spectator.co.uk ->
    spectator); on hosting platforms the subdomain; a trailing "online" is not part of the name (spiked-online -> spiked)."""
    import tldextract
    e = tldextract.TLDExtract(suffix_list_urls=())(domain)
    name = e.subdomain.split(".")[-1] if e.domain in PLATFORMS and e.subdomain else e.domain
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    return [name] + ([name[:-6]] if name.endswith("online") and len(name) > 9 else [])


def common_words(names, texts, domains):
    """Names (and names without "the") that are also common words: lowercase, as a whole word, more than
    COMMON_WORD_COUNT times in the articles of the other outlets (outside links and accounts)."""
    counts = collections.defaultdict(collections.Counter)                # outlet -> word -> count
    for d, t in zip(domains, _without_links(texts)):
        counts[d].update(re.findall(r"[A-Za-z0-9]+", t))
    total = sum(counts.values(), collections.Counter())
    common = set()
    for d, names_d in names.items():
        for name in names_d:
            for n in {name, name[3:] if name.startswith("the") else name}:
                if len(n) >= 2 and total[n] - counts[d][n] > COMMON_WORD_COUNT:
                    common.add(n)
    return common


def lowercase_names(names, common, texts, domains):
    """Common-word names that the outlet itself writes in lowercase in >= LOWERCASE_SHARE of its articles."""
    texts, domains = _without_links(texts).reset_index(drop=True), pd.Series(domains).reset_index(drop=True)
    own = set()
    for d, names_d in names.items():
        for n in names_d:
            if n in common and texts[domains == d].str.contains(re.compile(WORD_EDGE_L + re.escape(n) + WORD_EDGE_R)).mean() >= LOWERCASE_SHARE:
                own.add(n)
    return own


def name_pattern(names, common, lowercase=frozenset()):
    """The regular expression that finds the outlet names.
    - distinctive name (bellingcat): any case, joined or spaced ("The American Conservative"), with or without "the";
    - common-word name (politics): only capitalised (Politics, POLITICS), unless the outlet writes it in lowercase;
    - name starting with "the" (theweek): "the" may be dropped only if the rest is not a common word."""
    spaced = lambda n: r"[\s\-]?".join(map(re.escape, n))
    capital = lambda n: f"(?-i:{re.escape(n[0].upper())})" + "".join(r"[\s\-]?" + re.escape(c) for c in n[1:])
    forms = []
    for name in names:
        rest = name[3:] if name.startswith("the") and len(name) > 3 else None
        if name in common and name not in lowercase:
            forms.append(capital(name))
        elif rest and rest in common:
            forms.append(capital("the") + r"[\s\-]?" + capital(rest))
        else:
            forms += [r"(?:the[\s\-]?)?" + spaced(n) for n in [name] + ([rest] if rest else [])]
    return re.compile(WORD_EDGE_L + "(?:" + "|".join(forms) + ")" + WORD_EDGE_R, re.I)


# ------------------------------------------------------------------------- 2. the title signature
def title_tail(title):
    """The last part of a title after " | ", " – ", " — " or " - "."""
    m = re.search(r"\s[|–—-]\s([^|–—]{2,60})$", str(title))
    return m.group(1).strip() if m else None


def signatures(art):
    """Per outlet, the title tail repeated in >= SIGNATURE_SHARE of its titles (and >= 3). It finds names that the
    domain does not give (cfr.org -> "Council on Foreign Relations"). A one-word signature that is a common word
    ("Signed") is not a name and is dropped."""
    texts, domains = _without_links(art.text).reset_index(drop=True), art.domain.reset_index(drop=True)
    out = {}
    for d, g in art.groupby("domain"):
        tails = g.title.map(title_tail).value_counts()
        if len(tails) and tails.iloc[0] >= max(3, SIGNATURE_SHARE * len(g)):
            s = tails.index[0]
            word = re.compile(WORD_EDGE_L + re.escape(s.lower()) + WORD_EDGE_R)
            if " " not in s and texts[domains != d].str.count(word).sum() > COMMON_WORD_COUNT:
                continue
            out[d] = s
    return out


def signature_pattern(signature):
    """The signature, any case, up to the end of its last word ("Frontpage Mag" also finds "FrontPage Magazine")."""
    return re.compile(r"(?<![\w])" + re.escape(signature) + r"\w*", re.I) if signature else None


# ------------------------------------------------------------------------- 3. the authors
def own_authors(art):
    """Per outlet, the authors who sign only there and look like a person: 1-4 capitalised words, no digits, no words
    such as Press, Staff, Editor. Authors of several outlets (agencies, "Staff") do not identify the outlet."""
    rows = []
    for d, a in zip(art.domain, art.author.fillna("")):
        for name in re.split(r"\s*(?:;|,|\||&|\band\b)\s*", str(a)):
            name = re.sub(r"^\s*by\s+", "", name.strip(), flags=re.I).strip(" .")
            words = name.split()
            if (1 <= len(words) <= 4 and len(name) >= 3 and all(re.fullmatch(r"[A-ZÀ-Ý][\w.'’-]*", w) for w in words)
                    and not any(w.lower().strip(".") in NOT_A_PERSON for w in words)):
                rows.append((d, name))
    r = pd.DataFrame(rows, columns=["domain", "author"]).drop_duplicates()
    r = r[r.author.map(r.groupby("author").domain.nunique()) == 1]
    return r.groupby("domain").author.apply(lambda s: sorted(s, key=len, reverse=True)).to_dict()


def author_pattern(authors):
    """Exact author names (case included), longest first: 2-4 words anywhere, one word only after "by"."""
    full = [n for n in authors if " " in n]
    single = [n for n in authors if " " not in n]
    forms = []
    if full:
        forms.append(r"(?<![\w])(?:" + "|".join(map(re.escape, full)) + r")(?![\w])")
    if single:
        forms.append(r"(?<=\b[Bb]y )(?:" + "|".join(map(re.escape, single)) + r")(?![\w])")
    return re.compile("|".join(forms)) if forms else None


# ------------------------------------------------------------------------- 4. all rules together
def anonymize_text(text, name_re, author_re, signature_re):
    """The six rules, in order. Returns the text and how many replacements each rule made."""
    text = text if isinstance(text, str) else ""
    counts = {}
    for rule, pattern, token in [("signature", signature_re, "[OUTLET]"), ("email", EMAIL, "[EMAIL]"),
                                 ("url", URL, "[URL]"), ("account", ACCOUNT, "[ACCOUNT]"),
                                 ("author", author_re, "[AUTHOR]"), ("name", name_re, "[OUTLET]")]:
        if pattern is None:
            counts[rule] = 0
        else:
            text, counts[rule] = pattern.subn(token, text)
    return text, counts


def anonymize(art):
    """Anonymize text and title of every article. Returns the anonymous articles, the replacements per article and
    what the rules found (names, common words, lowercase names, signatures, authors)."""
    names = {d: outlet_names(d) for d in sorted(art.domain.unique())}
    common = common_words(names, art.text, art.domain)
    lower = lowercase_names(names, common, art.text, art.domain)
    sigs = signatures(art)
    authors = own_authors(art)
    new, counts = art.copy(), []
    for d, g in art.groupby("domain", sort=False):
        name_re, author_re, sig_re = name_pattern(names[d], common, lower), author_pattern(authors.get(d, [])), signature_pattern(sigs.get(d))
        for i, r in g.iterrows():
            new.at[i, "text"], c = anonymize_text(r.text, name_re, author_re, sig_re)
            new.at[i, "title"], ct = anonymize_text(r.title, name_re, author_re, sig_re)
            counts.append({"article_id": r.article_id, "domain": d, **c, **{f"title_{k}": v for k, v in ct.items()}})
    found = {"names": names, "common": common, "lowercase": lower, "signatures": sigs, "authors": authors}
    return new, pd.DataFrame(counts), found


# ------------------------------------------------------------------------- 5. automatic checks
def leftovers(art, found):
    """After anonymization, per outlet: in how many articles (text or title) its names or signature are still there,
    as a whole word and also inside other words. For common-word names the lowercase word stays by design."""
    rows = []
    for d, g in art.groupby("domain"):
        text = g.text.astype(str) + "\n" + g.title.astype(str)
        names = found["names"][d]
        look = [re.escape(n) for n in names] + ([re.escape(found["signatures"][d])] if d in found["signatures"] else [])
        common = any((n in found["common"] or n[3:] in found["common"]) and n not in found["lowercase"] for n in names)
        rows.append({"domain": d, "names": " / ".join(names), "signature": found["signatures"].get(d, ""),
                     "common_word": common, "articles": len(g),
                     "left_as_word": int(text.str.contains(re.compile(WORD_EDGE_L + "(?:" + "|".join(look) + ")" + WORD_EDGE_R, re.I)).sum()),
                     "left_inside_words": int(text.str.contains(re.compile("|".join(look), re.I)).sum())})
    return pd.DataFrame(rows)


def examples(texts, pattern, n=5, width=60, seed=SEED):
    """A few random snippets around a match, to see what was removed or what is left."""
    found = [str(t)[max(0, m.start() - width): m.end() + width].replace("\n", " ") for t in texts for m in pattern.finditer(str(t))]
    if not found:
        return []
    return [found[i] for i in np.random.default_rng(seed).choice(len(found), size=min(n, len(found)), replace=False)]
