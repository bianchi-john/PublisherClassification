"""Misure di stile di un articolo (con spaCy) e il testo "senza argomento" (text distortion, Stamatatos 2017).

Usato da 02_mappe_e_stile.ipynb. Le liste di parole vengono dalle fonti citate; i lessici NRC (Mohammad e Turney 2013,
Mohammad 2018) si leggono dalla cartella dei lessici (non si ridistribuiscono).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Hyland (2005), Metadiscourse, appendice: hedge (cautela) e booster (enfasi), forme di una parola
CAUTELA = set("""about almost apparent apparently appear appeared appears approximately argue argued argues around assume
assumed broadly claim could couldn doubt doubtful essentially estimate estimated fairly feel feels felt frequently generally guess
indicate indicated indicates largely likely mainly may maybe might mostly often ought perhaps plausible plausibly possible possibly
postulate presumable presumably probable probably quite rather relatively roughly seem seemed seems should sometimes somewhat suggest
suggested suggests suppose supposed suspect suspects tend tended tends typical typically uncertain unclear unlikely usually would
wouldn""".split())
ENFASI = set("""actually always believe believed believes certain certainly clear clearly conclusively decidedly definite definitely
demonstrate demonstrated demonstrates doubtless establish established evident evidently find finds found incontestable
incontestably incontrovertible incontrovertibly indeed indisputable indisputably know known must never obvious obviously prove
proved proves realize realized realizes really show showed shown shows sure surely think thinks thought truly true undeniable
undeniably undisputedly undoubtedly""".split())
# connettivi del discorso (le forme di una parola più comuni, per tipo di relazione)
CAUSALI = {"because", "therefore", "thus", "hence", "consequently", "accordingly"}
AVVERSATIVI = {"however", "whereas", "nevertheless", "nonetheless", "instead", "conversely", "but"}
CONCESSIVI = {"although", "though", "despite", "notwithstanding"}
AGGIUNTIVI = {"moreover", "furthermore", "additionally", "likewise", "similarly"}
INTENSIFICATORI = {"very", "extremely", "totally", "absolutely", "completely", "utterly", "incredibly", "truly", "highly", "entirely"}
ATTRIBUZIONE = set("""say tell report state announce explain note add write testify confirm quote cite acknowledge insist warn
claim allege according reportedly allegedly source spokesperson spokesman spokeswoman statement interview""".split())
IO_NOI = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
TU_VOI = {"you", "your", "yours", "yourself", "yourselves"}
SUBORDINATE = {"advcl", "ccomp", "csubj", "acl", "relcl"}
ENTITA = {"PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "EVENT", "LAW", "PRODUCT", "WORK_OF_ART"}
NUMERI = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"}
NOMINALIZZAZIONE = re.compile(r"(?:tion|sion|ment|ness|ity|ance|ence|ism)s?$")

_nlp, _lessici = None, None


def leggi_lessici(cartella):
    """NRC EmoLex (parole con almeno un'emozione) e NRC VAD (il 10% di parole con l'attivazione più alta)."""
    cartella = Path(cartella)
    emo = pd.read_csv(cartella / "NRC-Emotion-Lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep="\t", names=["parola", "emozione", "v"])
    emozioni = set(emo[(emo.v == 1) & ~emo.emozione.isin(["positive", "negative"])].parola.astype(str).str.lower())
    vad = pd.read_csv(cartella / "NRC-VAD-Lexicon" / "OneFilePerDimension" / "arousal-NRC-VAD-Lexicon.txt", sep="\t", names=["parola", "a"])
    attivazione = set(vad[vad.a >= vad.a.quantile(0.9)].parola.astype(str).str.lower())
    return emozioni, attivazione


def _sillabe(parola):
    return max(1, len(re.findall(r"[aeiouy]+", parola.lower().rstrip("e"))))


def _profondita(frase):
    return max((sum(1 for _ in t.ancestors) for t in frase), default=0)


def misure(doc, emozioni, attivazione):
    """Circa 30 misure di stile di un articolo già analizzato da spaCy (per frase, per 100 o per 1000 parole)."""
    parole = [t for t in doc if t.is_alpha]
    n = max(len(parole), 1)
    frasi = [s for s in doc.sents if any(t.is_alpha for t in s)]
    ns = max(len(frasi), 1)
    lunghezze = [sum(t.is_alpha for t in s) for s in frasi] or [0]
    minuscole = [t.lower_ for t in parole]
    lemmi = [t.lemma_.lower() for t in parole]
    sillabe = sum(_sillabe(p) for p in minuscole)
    conta = lambda insieme: sum(p in insieme for p in minuscole)
    pos = pd.Series([t.pos_ for t in parole]).value_counts()
    quota_pos = lambda p: pos.get(p, 0) / n * 100
    ents = [e.label_ for e in doc.ents]
    testo = doc.text
    return {
        # leggibilità e lessico
        "parole per frase": n / ns,
        "variabilità della lunghezza delle frasi": float(np.std(lunghezze)),
        "sillabe per parola": sillabe / n,
        "leggibilità (Flesch)": 206.835 - 1.015 * n / ns - 84.6 * sillabe / n,
        "parole lunghe, 7+ lettere (%)": sum(len(p) >= 7 for p in minuscole) / n * 100,
        "varietà del lessico (TTR, 300 parole)": len(set(minuscole[:300])) / max(1, min(300, n)),
        "nominalizzazioni (%)": sum(t.pos_ == "NOUN" and bool(NOMINALIZZAZIONE.search(t.lower_)) for t in parole) / n * 100,
        # sintassi
        "profondità sintattica": float(np.mean([_profondita(s) for s in frasi])) if frasi else 0.0,
        "subordinate per frase": sum(t.dep_ in SUBORDINATE for t in doc) / ns,
        "distanza tra parole collegate": float(np.mean([abs(t.i - t.head.i) for t in parole if t.head is not t])) if n > 1 else 0.0,
        "frasi al passivo (%)": sum(any(t.dep_ in ("nsubjpass", "auxpass") for t in s) for s in frasi) / ns * 100,
        "aggettivi (%)": quota_pos("ADJ"), "avverbi (%)": quota_pos("ADV"), "verbi (%)": quota_pos("VERB"),
        "nomi propri (%)": quota_pos("PROPN"), "pronomi (%)": quota_pos("PRON"),
        # ragionamento e fonti (per 1000 parole)
        "connettivi causali (‰)": conta(CAUSALI) / n * 1000,
        "connettivi avversativi (‰)": conta(AVVERSATIVI) / n * 1000,
        "connettivi concessivi (‰)": conta(CONCESSIVI) / n * 1000,
        "connettivi aggiuntivi (‰)": conta(AGGIUNTIVI) / n * 1000,
        "cautela, hedge (‰)": conta(CAUTELA) / n * 1000,
        "enfasi, booster (‰)": conta(ENFASI) / n * 1000,
        "attribuzione alle fonti (‰)": sum(l in ATTRIBUZIONE for l in lemmi) / n * 1000,
        "virgolette (‰)": len(re.findall(r'["“”]', testo)) / n * 1000,
        "entità nominate (per 100 parole)": sum(e in ENTITA for e in ents) / n * 100,
        "numeri e date (per 100 parole)": sum(e in NUMERI for e in ents) / n * 100,
        # tono (per 1000 parole)
        "superlativi (‰)": sum(t.tag_ in ("JJS", "RBS") for t in parole) / n * 1000,
        "intensificatori (‰)": conta(INTENSIFICATORI) / n * 1000,
        "parole emotive (‰)": conta(emozioni) / n * 1000,
        "parole ad alta attivazione (‰)": conta(attivazione) / n * 1000,
        "io / noi (‰)": conta(IO_NOI) / n * 1000,
        "tu / voi (‰)": conta(TU_VOI) / n * 1000,
        "punti esclamativi (‰)": testo.count("!") / n * 1000,
        "punti di domanda (‰)": testo.count("?") / n * 1000,
        "parole in maiuscolo (%)": sum(t.text.isupper() and len(t.text) >= 3 for t in parole) / n * 100,
    }


def senza_argomento(doc):
    """Il testo con le sole parole funzionali e la punteggiatura; ogni altra parola diventa la sua categoria (NOUN, VERB...)."""
    return " ".join(t.lower_ if (t.is_stop or t.is_punct) else t.pos_ for t in doc if not t.is_space)


def analizza_blocco(testi, cartella_lessici, max_parole=3000):
    """Analizza un blocco di articoli (in un processo a parte): misure di stile e testo senza argomento."""
    global _nlp, _lessici
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        _nlp.max_length = 2_000_000
        _lessici = leggi_lessici(cartella_lessici)
    righe, distorti = [], []
    for doc in _nlp.pipe((" ".join(t.split()[:max_parole]) for t in testi), batch_size=32):
        righe.append(misure(doc, *_lessici))
        distorti.append(senza_argomento(doc))
    return righe, distorti
