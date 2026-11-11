"""Tentativo 13, fase 2 (XAI): dati, campione, categorie di parole e Integrated Gradients per i due modelli.

La quantità spiegata è la stessa per i due modelli: il **punteggio di affidabilità** f = log P(l3..l5) − log P(l1..l2)
(log del rapporto tra la probabilità di "affidabile" e di "non affidabile"). Le attribuzioni sono Integrated Gradients
(Sundararajan et al., 2017) con baseline "testo vuoto": per il TF-IDF il vettore nullo, per ModernBERT gli stessi token
speciali con [PAD] al posto delle parole. Per ogni articolo si usa il modello del fold in cui l'articolo era nel test.

Le parole si dividono in categorie con un ordine di precedenza (una parola va nella prima categoria che la riconosce),
da liste pubblicate dove esistono: entità di spaCy, Moral Foundations Dictionary 2.0 (Frimer et al., 2019), NRC Emotion
Lexicon (Mohammad e Turney, 2013), NRC VAD Lexicon (Mohammad, 2018; attivazione), hedge e booster di Hyland (2005),
termini di parte dal log-odds con prior di Dirichlet (Monroe et al., 2008) tra testate Right e Left, calcolato fold per
fold sulle sole testate di training.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

VERSIONI = Path("../../../../../Dataset/tentativi/13_XAI/versioni")
LESSICI = Path("../../../../../Dataset/risorse/lessici")
MODELLI = Path("../../../../../Modelli/13_XAI")
RESULTS = Path("../Results/3-xai")
RISULTATI_MODELLI = Path("../Results/2-modelli")
SEED = 42
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]
MAX_TOKEN = 1024

CATEGORIE = ["cornice del sito", "numeri e date", "nomi propri", "fonti e attribuzione", "rivolgersi al lettore",
             "cautela ed enfasi", "termini di parte", "morale e religione", "sensazionalismo", "emozioni",
             "punteggiatura", "parole funzione", "argomento", "altro"]

# ---------------------------------------------------------------- liste scritte qui (dalle fonti citate)
CORNICE = set("""subscribe subscribed subscriber subscribers subscription subscriptions unsubscribe newsletter newsletters
signup login logged click clicking donate donating donation donations sponsored advertisement advertisements advertise
advertising cookie cookies copyright getty paywall rss podcast podcasts membership premium share shares sharing tweet retweet
facebook twitter instagram telegram youtube tiktok gab rumble truthsocial whatsapp linkedin pinterest email emails""".split())
# Hyland (2005), Metadiscourse, appendice: hedge e booster (solo le forme di una parola)
CAUTELA_ENFASI = set("""about almost apparent apparently appear appeared appears approximately argue argued argues around assume
assumed broadly claim could couldn doubt doubtful essentially estimate estimated fairly feel feels felt frequently generally guess
indicate indicated indicates largely likely mainly may maybe might mostly often ought perhaps plausible plausibly possible possibly
postulate presumable presumably probable probably quite rather relatively roughly seem seemed seems should sometimes somewhat suggest
suggested suggests suppose supposed suspect suspects tend tended tends typical typically uncertain unclear unlikely usually would
wouldn actually always believe believed believes certain certainly clear clearly conclusively decidedly definite definitely
demonstrate demonstrated demonstrates doubtless establish established evident evidently find finds found incontestable
incontestably incontrovertible incontrovertibly indeed indisputable indisputably know known must never obvious obviously prove
proved proves realize realized realizes really show showed shown shows sure surely think thinks thought truly true undeniable
undeniably undisputedly undoubtedly""".split())
# verbi e parole di attribuzione (reporting verbs) e riferimenti alle fonti
ATTRIBUZIONE_LEMMI = set("""say tell report state announce explain note add write testify confirm quote cite acknowledge insist warn
claim allege according reportedly allegedly source spokesperson spokesman spokeswoman statement interview""".split())
LETTORE = set("i me my mine myself we us our ours ourselves you your yours yourself yourselves".split())
ENTITA = {"PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "EVENT", "LAW", "PRODUCT", "WORK_OF_ART"}
NUMERI = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"}
FUNZIONE_POS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PRON"}


def leggi_lessici():
    """MFD 2.0 (morale e religione), NRC EmoLex (emozioni), NRC VAD (attivazione: il 10% più alto = sensazionalismo)."""
    testo = (LESSICI / "mfd2.0.dic").read_text(encoding="utf-8", errors="ignore")
    righe = [r for r in re.split(r"\r\n|\r|\n", testo) if r.strip()]
    fine_intestazione = [i for i, r in enumerate(righe) if r.strip() == "%"][1]
    morale = {r.split("\t")[0].strip().lower() for r in righe[fine_intestazione + 1:] if "\t" in r}
    emo = pd.read_csv(LESSICI / "NRC-Emotion-Lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep="\t", names=["parola", "emozione", "v"])
    emozioni = set(emo[(emo.v == 1) & ~emo.emozione.isin(["positive", "negative"])].parola.str.lower())
    vad = pd.read_csv(LESSICI / "NRC-VAD-Lexicon" / "OneFilePerDimension" / "arousal-NRC-VAD-Lexicon.txt", sep="\t", names=["parola", "a"])
    soglia = vad.a.quantile(0.9)
    attivazione = set(vad[vad.a >= soglia].parola.astype(str).str.lower())
    return {"morale": morale, "emozioni": emozioni, "attivazione": attivazione, "soglia_attivazione": float(soglia)}


def categoria(tok, lessici, di_parte):
    """La categoria di un token spaCy (ordine di precedenza di CATEGORIE)."""
    t, l = tok.text.lower(), tok.lemma_.lower()
    if t in CORNICE or l in CORNICE:
        return "cornice del sito"
    if tok.like_num or tok.ent_type_ in NUMERI:
        return "numeri e date"
    if tok.ent_type_ in ENTITA:
        return "nomi propri"
    if l in ATTRIBUZIONE_LEMMI or t in ATTRIBUZIONE_LEMMI:
        return "fonti e attribuzione"
    if t in LETTORE:
        return "rivolgersi al lettore"
    if t in CAUTELA_ENFASI:
        return "cautela ed enfasi"
    if t in di_parte or l in di_parte:
        return "termini di parte"
    if t in lessici["morale"] or l in lessici["morale"]:
        return "morale e religione"
    if t == "!" or (tok.is_alpha and len(t) >= 3 and tok.text.isupper()) or t in lessici["attivazione"] or l in lessici["attivazione"]:
        return "sensazionalismo"
    if t in lessici["emozioni"] or l in lessici["emozioni"]:
        return "emozioni"
    if tok.is_punct:
        return "punteggiatura"
    if tok.is_stop or tok.pos_ in FUNZIONE_POS:
        return "parole funzione"
    if tok.pos_ in ("NOUN", "PROPN"):
        return "argomento"
    return "altro"


# ---------------------------------------------------------------- termini di parte (Monroe et al., 2008)
PAROLA = re.compile(r"[a-z][a-z'-]{2,}")


SCARTI = set("com www http https html htm amp org net pdf jpg png".split())


def termini_di_parte(art, testate, fold, n=200, z_min=3.0, max_articoli=50, min_testate=10):
    """Le parole più tipiche delle testate Right contro le Left (e viceversa), dalle sole testate di training del fold.

    Solo parole piene (via le stopword di spaCy e i residui di indirizzi web) usate da almeno min_testate testate
    diverse (così non entrano le parole di un solo sito); al più n per lato, con |z| > z_min."""
    from spacy.lang.en.stop_words import STOP_WORDS
    tr = testate[(testate.fold_gruppo != fold) & testate.orientamento.isin(["Right", "Left"])]
    a = art[art.domain.isin(tr.index)]
    a = a.sample(frac=1, random_state=SEED); a = a[a.groupby("domain").cumcount() < max_articoli]     # al più max_articoli a caso per testata
    conta = {o: Counter() for o in ["Right", "Left"]}
    testate_di = {}
    for d, testo in zip(a.domain, a.text):
        parole = [w for w in PAROLA.findall(testo.lower()) if w not in STOP_WORDS and w not in SCARTI]
        conta[tr.orientamento[d]].update(parole)
        for w in set(parole):
            testate_di.setdefault(w, set()).add(d)
    voc = [w for w in set(conta["Right"]) | set(conta["Left"])
           if conta["Right"][w] + conta["Left"][w] >= 20 and len(testate_di.get(w, ())) >= min_testate]
    yR = np.array([conta["Right"][w] for w in voc], float); yL = np.array([conta["Left"][w] for w in voc], float)
    prior = (yR + yL) / (yR + yL).sum() * 1000                        # prior di Dirichlet informativo, somma 1000
    nR, nL, a0 = yR.sum(), yL.sum(), prior.sum()
    delta = np.log((yR + prior) / (nR + a0 - yR - prior)) - np.log((yL + prior) / (nL + a0 - yL - prior))
    z = delta / np.sqrt(1 / (yR + prior) + 1 / (yL + prior))
    ordine = np.argsort(z)
    destra = [voc[i] for i in ordine[::-1][:n] if z[i] > z_min]
    sinistra = [voc[i] for i in ordine[:n] if z[i] < -z_min]
    return set(destra) | set(sinistra), destra[:30], sinistra[:30]


# ---------------------------------------------------------------- campione da spiegare
def campione(art, testate, per_livello=60, per_testata=3):
    """Articoli delle testate valutate, bilanciati per livello: al più per_testata articoli per testata, a giro."""
    rng = np.random.default_rng(SEED)
    val = testate[testate.valutazione]
    scelti = []
    for k in range(5):
        doms = list(val.index[val.label == k]); rng.shuffle(doms)
        per_dom = {d: list(rng.permutation(art.index[art.domain == d].to_numpy())) for d in doms}
        presi = []
        for giro in range(per_testata):
            for d in doms:
                if len(presi) < per_livello and len(per_dom[d]) > giro:
                    presi.append(per_dom[d][giro])
        scelti += presi
    return art.loc[scelti]


# ---------------------------------------------------------------- Integrated Gradients
def punteggio(z):
    """f = log P(l3..l5) − log P(l1..l2) dai logit (ultima dimensione: le 5 classi)."""
    from scipy.special import logsumexp
    return logsumexp(z[..., 2:], axis=-1) - logsumexp(z[..., :2], axis=-1)


def ig_feature(m, testo, passi=50):
    """IG sul TF-IDF + regressione logistica: attribuzione di ogni n-gramma, poi distribuita sui token del testo.

    Ritorna (lista di (inizio, fine, attribuzione) per i token di parola del vettorizzatore, f(x), f(0))."""
    from sklearn.feature_extraction.text import strip_accents_unicode
    v, clf = m["tfidf"], m["modello"]
    x = v.transform([testo]); idx = x.indices; xv = x.data
    W = np.zeros((5, len(idx))); b = np.zeros(5)
    W[clf.classes_] = clf.coef_[:, idx]; b[clf.classes_] = clf.intercept_
    grad = np.zeros(len(idx))
    for a in (np.arange(passi) + 0.5) / passi:
        z = a * (W @ xv) + b
        p = np.exp(z - z.max()); p /= p.sum()
        g = np.concatenate([-p[:2] / p[:2].sum(), p[2:] / p[2:].sum()])
        grad += g @ W
    attr_feat = dict(zip(idx, xv * grad / passi))
    f_x, f_0 = float(punteggio(W @ xv + b)), float(punteggio(b))
    # i token come li vede il vettorizzatore, con le posizioni nel testo
    token = [(mt.start(), mt.end(), strip_accents_unicode(mt.group().lower())) for mt in re.finditer(r"(?u)\b\w\w+\b", testo)]
    voc = v.vocabulary_
    occorrenze = Counter()
    pezzi = []
    for i, (s, e, w) in enumerate(token):
        candidati = [(w, [(s, e)])]                                   # la parola
        if i + 1 < len(token):                                        # e la coppia con la parola dopo
            s2, e2, w2 = token[i + 1]
            candidati.append((w + " " + w2, [(s, e), (s2, e2)]))
        for ngram, span in candidati:
            if ngram in voc and voc[ngram] in attr_feat:
                occorrenze[voc[ngram]] += 1
                pezzi.append((voc[ngram], span))
    out = []
    for j, span in pezzi:
        quota = attr_feat[j] / occorrenze[j] / len(span)
        out += [(s, e, quota) for s, e in span]
    return out, f_x, f_0


def ig_modernbert(model, tok, testo, device, passi=64, lotto=4):
    """IG su ModernBERT, rispetto agli embedding dei token (captum), in precisione piena (fp32); in più, come controllo,
    gradiente × input (un solo passaggio). Ritorna ((inizio, fine, attribuzione IG, attribuzione gradiente × input) per
    token, f(x), f(base), delta di convergenza, fine della parte letta in caratteri)."""
    import torch
    from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
    enc = tok(testo, truncation=True, max_length=MAX_TOKEN, return_offsets_mapping=True, return_tensors="pt")
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    speciali = torch.tensor(tok.get_special_tokens_mask(enc["input_ids"][0].tolist(), already_has_special_tokens=True), device=device).bool()
    base = torch.where(speciali, ids, torch.full_like(ids, tok.pad_token_id))

    def f(input_ids, attention_mask):
        z = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        return torch.logsumexp(z[:, 2:], -1) - torch.logsumexp(z[:, :2], -1)

    lig = LayerIntegratedGradients(f, model.model.embeddings.tok_embeddings)
    attr, delta = lig.attribute(ids, baselines=base, additional_forward_args=(mask,), n_steps=passi, internal_batch_size=lotto,
                                return_convergence_delta=True)
    a = attr.sum(-1)[0].detach().float().cpu().numpy()
    gxi = LayerGradientXActivation(f, model.model.embeddings.tok_embeddings).attribute(ids, additional_forward_args=(mask,))
    g = gxi.sum(-1)[0].detach().float().cpu().numpy()
    with torch.no_grad():
        fx, fb = float(f(ids, mask)), float(f(base, mask))
    off = enc["offset_mapping"][0].tolist(); sp = speciali.cpu().numpy()
    out = [(s, e, float(a[i]), float(g[i])) for i, (s, e) in enumerate(off) if not sp[i] and e > s]
    fine_finestra = max((e for s, e in off), default=0)
    return out, fx, fb, float(delta), fine_finestra


def su_token(inizi, fini, pezzi):
    """Somma le attribuzioni (inizio, fine, valore) sui token (posizioni inizi/fini) che si sovrappongono, in proporzione ai caratteri."""
    inizi, fini = np.asarray(inizi), np.asarray(fini)
    v = np.zeros(len(inizi))
    for s, e, a in pezzi:
        i = int(np.searchsorted(fini, s, side="right"))
        tocchi = []
        while i < len(inizi) and inizi[i] < e:
            sovr = min(e, fini[i]) - max(s, inizi[i])
            if sovr > 0:
                tocchi.append((i, sovr))
            i += 1
        tot = sum(o for _, o in tocchi)
        for j, o in tocchi:
            v[j] += a * o / tot
    return v


def togli(testo, spans):
    """Il testo senza i pezzi indicati ((inizio, fine) in caratteri), con gli spazi doppi ridotti."""
    parti, ultimo = [], 0
    for s, e in sorted(spans):
        if s < ultimo:
            continue
        parti.append(testo[ultimo:s]); ultimo = e
    parti.append(testo[ultimo:])
    return re.sub(r"[ \t]{2,}", " ", "".join(parti))
