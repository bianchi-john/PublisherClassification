"""Builds Attempt20.odp in John's style (from his Regressione.odp: black title with a red part, green subtitle, short
bullet lists, the index repeated with the current section in red). Usage: python build_slides.py"""
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "Criteri e regressione" / "Regressione.odp"
OUT = HERE / "Attempt20.odp"

# numbers checked on the data (steps 1-5 of attempt 20)
N = {"tried": "5,113", "scraped": "264,203", "outlets": "226", "articles": "25,524", "touched": "54.6%",
     "max_tokens": "3,000"}

z = zipfile.ZipFile(TEMPLATE)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()
c = c.replace("</office:automatic-styles>", """<style:style style:name="X20" style:family="paragraph"><style:text-properties fo:font-size="20pt" style:font-size-asian="20pt" style:font-size-complex="20pt"/></style:style>
<style:style style:name="X24" style:family="paragraph"><style:text-properties fo:font-size="24pt" style:font-size-asian="24pt" style:font-size-complex="24pt"/></style:style>
<style:style style:name="Xb" style:family="text"><style:text-properties fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"/></style:style>
<style:style style:name="Xr" style:family="text"><style:text-properties fo:color="#c9211e" loext:opacity="100%" fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"/></style:style>
<style:style style:name="Xg" style:family="text"><style:text-properties fo:color="#00a933" loext:opacity="100%" fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"/></style:style>
<style:style style:name="Xm" style:family="text"><style:text-properties style:font-name="Liberation Mono" fo:font-family="'Liberation Mono'"/></style:style>
</office:automatic-styles>""", 1)


def runs(text):
    """**bold**, [[red]], {{green}}, `mono` -> spans."""
    out, pos = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|\[\[(.+?)\]\]|\{\{(.+?)\}\}|`(.+?)`", text):
        out.append(escape(text[pos:m.start()]))
        style, t = [(s, g) for s, g in zip(["Xb", "Xr", "Xg", "Xm"], m.groups()) if g is not None][0]
        out.append(f'<text:span text:style-name="{style}">{escape(t)}</text:span>')
        pos = m.end()
    return "".join(out) + escape(text[pos:])


def bullets(items, p, list_style="L2"):
    xml = f'<text:list text:style-name="{list_style}">' if list_style else "<text:list>"
    for it in items:
        text, children = (it, []) if isinstance(it, str) else it
        xml += f'<text:list-item><text:p text:style-name="{p}">{runs(text)}</text:p>'
        xml += bullets(children, p, None) if children else ""
        xml += "</text:list-item>"
    return xml + "</text:list>"


def notes(n):
    return (f'<presentation:notes draw:style-name="dp2"><draw:page-thumbnail draw:style-name="gr1" draw:layer="layout" '
            f'svg:width="18.624cm" svg:height="10.476cm" svg:x="1.482cm" svg:y="2.123cm" draw:page-number="{n}" '
            f'presentation:class="page"/><draw:frame presentation:style-name="pr2" draw:text-style-name="P1" '
            f'draw:layer="layout" svg:width="17.271cm" svg:height="12.572cm" svg:x="2.159cm" svg:y="13.271cm" '
            f'presentation:class="notes" presentation:placeholder="true"><draw:text-box/></draw:frame></presentation:notes>')


def page(black, red, green, items, p="X20"):
    """Like John's slide 5: title (black + red), green subtitle, bullet list."""
    title = (f'<draw:frame presentation:style-name="pr1" draw:layer="layout" svg:width="25.199cm" svg:height="2.629cm" '
             f'svg:x="1.4cm" svg:y="0cm" presentation:class="title" presentation:user-transformed="true"><draw:text-box>'
             f'<text:p>{escape(black)}<text:span text:style-name="T4">{escape(red)}</text:span></text:p></draw:text-box></draw:frame>')
    sub = (f'<draw:frame presentation:style-name="pr4" draw:text-style-name="P5" draw:layer="layout" svg:width="25.199cm" '
           f'svg:height="1.102cm" svg:x="1.471cm" svg:y="2.073cm" presentation:class="title" presentation:user-transformed="true">'
           f'<draw:text-box><text:p text:style-name="P4"><text:span text:style-name="T8">{escape(green)}</text:span></text:p>'
           f'</draw:text-box></draw:frame>')
    body = (f'<draw:frame presentation:style-name="pr3" draw:layer="layout" svg:width="25.199cm" svg:height="11.6cm" '
            f'svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline" presentation:user-transformed="true">'
            f'<draw:text-box>{bullets(items, p)}</draw:text-box></draw:frame>')
    return (f'<draw:page draw:name="NAME" draw:style-name="dp1" draw:master-page-name="Default" '
            f'presentation:presentation-page-layout-name="AL2T1">{title}{body}{sub}NOTES</draw:page>')


SECTIONS = ["Data", "Models", "Evaluation"]


def index(active):
    """John's index slide: numbered list, the current item in bold red."""
    items = "".join(f'<text:list-item><text:p>{f"<text:span text:style-name=\"T3\">{s}</text:span>" if i == active else s}'
                    f'</text:p></text:list-item>' for i, s in enumerate(SECTIONS))
    return (f'<draw:page draw:name="NAME" draw:style-name="dp1" draw:master-page-name="Default" '
            f'presentation:presentation-page-layout-name="AL2T1"><draw:frame presentation:style-name="pr1" draw:layer="layout" '
            f'svg:width="25.199cm" svg:height="2.629cm" svg:x="1.4cm" svg:y="0.628cm" presentation:class="title" '
            f'presentation:user-transformed="true"><draw:text-box><text:p>Attempt 20, step by step</text:p></draw:text-box>'
            f'</draw:frame><draw:frame presentation:style-name="pr3" draw:layer="layout" svg:width="25.199cm" svg:height="9.134cm" '
            f'svg:x="1.4cm" svg:y="3.685cm" presentation:class="outline" presentation:user-transformed="true"><draw:text-box>'
            f'<text:list text:style-name="L3">{items}</text:list></draw:text-box></draw:frame>NOTES</draw:page>')


original = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
title_page = original[0].replace("Analisi Regressione", "Attempt 20: from scratch")

slides = [
    title_page,
    page("Attempt 20: ", "from scratch", "What we want to know", [
        "Can the **text alone** tell how reliable an outlet is? (NewsGuard 2026: score 0-100, 5 levels)",
        "On outlets **never seen** in training",
        "With **no trace of who wrote it**: no name, no authors, no links",
        "Is it better to predict the **level** (classification) or the **exact score** (regression)?",
        ("Everything redone in one clean pipeline", ["code and comments in English", "every step reproducible (fixed seeds)"]),
    ], p="X24"),
    index(0),
    page("Step 1: ", "scraping", "Not re-run: we use the October 2026 scraping", [
        f"{N['tried']} political NewsGuard outlets tried, in two rounds",
        "Up to 200 articles per outlet, from feeds, sitemaps and homepage",
        f"{N['scraped']} articles over 100 words, with their **raw HTML** saved",
        "Same code, translated: `scraper.py` + `run_scraper.sh`",
    ], p="X24"),
    page("Step 2: ", "cleaning", "Simple rules, fixed before the results", [
        "Only **political** outlets (a single NewsGuard topic) with a score",
        "Text re-extracted from the HTML with trafilatura (**precision** mode)",
        "English, 100-1,500 words, published from 2025",
        "No **duplicates** or near duplicates, across all outlets",
        "At least 10 articles per outlet",
        "**One outlet per ownership group** (same NewsGuard Owner or Parent Domain)",
        (f"Result: **{N['outlets']} outlets, {N['articles']} articles**", ["the same dataset as attempt 16"]),
    ]),
    page("Step 3: ", "anonymization", "Who published it must disappear", [
        ("Six rules, in this order", ["outlet signature in the titles (\"... | The New York Sun\") -> `[OUTLET]`",
                                      "e-mails -> `[EMAIL]`, links -> `[URL]`, @accounts -> `[ACCOUNT]`",
                                      "the outlet's own authors -> `[AUTHOR]`",
                                      "the outlet name from its domain -> `[OUTLET]`"]),
        "Example: \"Sign up to receive [OUTLET]'s biggest investigations\"",
        f"{N['touched']} of the articles change; people, places and facts stay",
        "Automatic check: **no outlet name left** as a word",
    ]),
    index(1),
    page("Step 4: ", "split", "One split for both tasks", [
        "Outlets sorted by **exact score**, in blocks of 5: each one to a different fold",
        "Every fold: same score distribution, same number of outlets per level (±1)",
        "An outlet is **never** split between training and test",
        "Same outlets in the same folds -> classifier and regressor compared **in pairs**",
    ], p="X24"),
    page("Steps 5-7: ", "the models", "Classification and regression, same models", [
        ("Four ways to read the text", ["**TF-IDF** + logistic regression / ridge",
                                        "**Qwen3-Embedding-8B**, frozen + linear model",
                                        "**Mistral-Small-24B**, frozen + linear model",
                                        "**ModernBERT**, fully trained"]),
        f"Every article read **in full** (limit 4,096 tokens, longest article {N['max_tokens']})",
        "Classification: the 5 levels. Regression: the exact score 0-100",
        "Saved: logits / score of **every article of every fold**, and **every fold model**",
    ]),
    index(2),
    page("Steps 8-9: ", "evaluation", "Two approaches, two spaces", [
        ("**Levels** (F1, tolerant ±1, kappa, Spearman)", ["classifier: majority level of the articles",
                                                           "regressor: NewsGuard bounds, training quotas or article vote"]),
        ("**Score** (MAE, Spearman, within 10 / 20 points)", ["regressor: mean score, or remapped to the training scores",
                                                              "classifier: expected score from its probabilities"]),
        "Baseline without text: most frequent level, mean training score",
        "Per outlet, 95% intervals (bootstrap), classifier vs regressor **in pairs**",
    ]),
    page("", "Where we are", "Built, checked, running", [
        "Steps 1-4 run and checked: same dataset as attempt 16",
        "Steps 5-9 running on the server (about one day)",
        "All results on one page: `09_results.ipynb`",
    ], p="X24"),
]

body = ""
for n, s in enumerate(slides, 1):
    s = re.sub(r'draw:name="[^"]*"', f'draw:name="page{n}"', s, count=1)
    s = s.replace("NOTES", notes(n))
    body += re.sub(r'draw:page-number="\d+"', f'draw:page-number="{n}"', s)
start, end = c.find("<draw:page "), c.rfind("</draw:page>") + len("</draw:page>")
c = c[:start] + body + c[end:]

with zipfile.ZipFile(OUT, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
    for info in z.infolist():
        if info.filename == "mimetype":
            continue
        data = c.encode() if info.filename == "content.xml" else z.read(info.filename)
        zo.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
print(len(slides), "slides ->", OUT)
