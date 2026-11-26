"""SUPERATO dal 5 novembre: per le revisioni successive si usa aggiorna_regressori.py (con stile_slide.py).
Aggiorna FinalState.odp (il file modificato a mano da John) senza toccare le sue modifiche: cambia solo i punti
indicati dai suoi commenti, toglie i commenti risolti e aggiunge slide nuove nel suo stile (titolo rosso, sottotitolo
verde, testo 24 o 20 pt, un punto per clic). Le slide nuove usano gli stili già presenti nel suo file.
Uso:  python aggiorna_slide.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

QUI = Path(__file__).resolve().parent
FIG = QUI / "figure"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
z = zipfile.ZipFile(IN)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()
for s in ["T3", "T4", "T5", "T6", "T7", "T8", "T10", "T11", "T13", "T15", "L2", "pr1", "pr2", "pr4", "P5", "gr2", "dp1"]:
    assert f'style:name="{s}"' in c, s
STILI = {24: {"": "T3", "b": "T4", "r": "T15", "g": "T8"}, 20: {"": "T13", "b": "T6", "r": "T7", "g": "T8"}}
immagini = {}
contatore = [0]


# ------------------------------------------------------------------------------------------------ slide nuove, nel suo stile
def runs(testo, corpo):
    """**grassetto**, [[rosso]], ((grigio)) → gli stili di John per quella dimensione del testo."""
    st, out, pos = STILI[corpo], [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|\[\[(.+?)\]\]|\(\((.+?)\)\)", testo):
        if m.start() > pos:
            out.append(f'<text:span text:style-name="{st[""]}">{escape(testo[pos:m.start()])}</text:span>')
        k, t = [(k, g) for k, g in zip("brg", m.groups()) if g is not None][0]
        out.append(f'<text:span text:style-name="{st[k]}">{escape(t)}</text:span>')
        pos = m.end()
    if pos < len(testo):
        out.append(f'<text:span text:style-name="{st[""]}">{escape(testo[pos:])}</text:span>')
    return "".join(out)


def elenco(voci, corpo, ids, principale=True):
    xml = '<text:list text:style-name="L2">' if principale else "<text:list>"
    for v in voci:
        testo, figli = (v, []) if isinstance(v, str) else v
        contatore[0] += 1
        pid = f"zp{contatore[0]}"
        ids.append(pid)
        xml += f'<text:list-item><text:p xml:id="{pid}" text:id="{pid}">{runs(testo, corpo)}</text:p>'
        xml += elenco(figli, corpo, ids, False) if figli else ""
        xml += "</text:list-item>"
    return xml + "</text:list>"


def animazione(ids):
    passi = "".join(
        f'<anim:par smil:begin="next" smil:fill="hold"><anim:par smil:begin="0s" smil:fill="hold"><anim:par smil:begin="0s" '
        f'smil:fill="hold" presentation:node-type="on-click" presentation:preset-class="entrance" '
        f'presentation:preset-id="ooo-entrance-appear"><anim:set smil:begin="0s" smil:dur="0.001s" smil:fill="hold" '
        f'smil:targetElement="{i}" smil:attributeName="visibility" smil:to="visible"/></anim:par></anim:par></anim:par>' for i in ids)
    return (f'<anim:par smil:dur="indefinite" smil:restart="never" presentation:node-type="timing-root"><anim:seq '
            f'smil:dur="indefinite" presentation:node-type="main-sequence">{passi}</anim:seq></anim:par>') if ids else ""


def immagine(percorso, x, y, largo=None, alto=None):
    """Un'immagine: data la larghezza o l'altezza (cm), l'altra segue le proporzioni."""
    w, h = Image.open(percorso).size
    if largo is None:
        largo = alto * w / h
    alto = largo * h / w
    nome = f"Pictures/z_{percorso.parent.name}_{percorso.name}"
    immagini[nome] = percorso
    return (f'<draw:frame draw:style-name="gr2" draw:layer="layout" svg:width="{largo:.2f}cm" svg:height="{alto:.2f}cm" '
            f'svg:x="{x:.2f}cm" svg:y="{y:.2f}cm"><draw:image xlink:href="{nome}" xlink:type="simple" xlink:show="embed" '
            f'xlink:actuate="onLoad" draw:mime-type="image/png"><text:p/></draw:image></draw:frame>')


def figura(percorso, y, largo_max=25.0, basso=15.3):
    """Un grafico centrato sotto il testo, il più grande possibile tra `y` e il fondo della slide."""
    w, h = Image.open(percorso).size
    largo = min(largo_max, (basso - y) * w / h)
    return immagine(percorso, (28 - largo) / 2, y, largo=largo)


def pagina(titolo, sottotitolo, voci=(), corpo=24, alto_testo=None, extra=""):
    """Titolo rosso, sottotitolo verde, elenco (corpo 24 o 20 pt), eventuali immagini in `extra`."""
    ids = []
    alto_testo = alto_testo or 11.6
    t = (f'<draw:frame presentation:style-name="pr1" draw:layer="layout" svg:width="25.199cm" svg:height="2.629cm" '
         f'svg:x="1.4cm" svg:y="0cm" presentation:class="title" presentation:user-transformed="true"><draw:text-box>'
         f'<text:p><text:span text:style-name="T5">{escape(titolo)}</text:span></text:p></draw:text-box></draw:frame>')
    s = (f'<draw:frame presentation:style-name="pr4" draw:text-style-name="P5" draw:layer="layout" svg:width="25.199cm" '
         f'svg:height="1.102cm" svg:x="1.471cm" svg:y="2.073cm" presentation:class="title" presentation:user-transformed="true">'
         f'<draw:text-box><text:p><text:span text:style-name="T11">{escape(sottotitolo)}</text:span></text:p></draw:text-box>'
         f'</draw:frame>') if sottotitolo else ""
    b = (f'<draw:frame presentation:style-name="pr2" draw:layer="layout" svg:width="25.199cm" svg:height="{alto_testo:.2f}cm" '
         f'svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline" presentation:user-transformed="true"><draw:text-box>'
         f'{elenco(voci, corpo, ids)}</draw:text-box></draw:frame>') if voci else ""
    return (f'<draw:page draw:name="NOME" draw:style-name="dp1" draw:master-page-name="Default" '
            f'presentation:presentation-page-layout-name="AL1T1">{t}{b}{extra}{s}{animazione(ids)}</draw:page>')


def con_grafico(titolo, sottotitolo, voci, grafico, corpo=20, y=None, largo_max=25.0):
    """Testo breve in alto, grafico sotto."""
    n = sum(1 + (len(v[1]) if not isinstance(v, str) else 0) for v in voci)
    y = y or (3.6 + 0.95 * n + 0.3 if voci else 3.4)
    return pagina(titolo, sottotitolo, voci, corpo, alto_testo=max(y - 3.4, 0.8), extra=figura(grafico, y, largo_max))


def con_codice(titolo, sottotitolo, voci, codici, corpo=20, y=None):
    """Testo breve in alto, una o due schermate di codice sotto (una sotto l'altra)."""
    n = sum(1 + (len(v[1]) if not isinstance(v, str) else 0) for v in voci)
    y = y or (3.6 + 0.95 * n + 0.3)
    extra, spazio = "", (15.3 - y) / len(codici)
    for i, cod in enumerate(codici):
        extra += figura(FIG / "codice" / f"{cod}.png", y + i * spazio, basso=y + (i + 1) * spazio - 0.2)
    return pagina(titolo, sottotitolo, voci, corpo, alto_testo=y - 3.4, extra=extra)


V, M = FIG / "valutazione", FIG / "modelli"

# ------------------------------------------------------------------------------------------------ le slide di John
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(pagine) == 17, len(pagine)
titoli = [" ".join(re.sub(r"<[^>]+>", " ", re.search(r'presentation:class="title".*?</draw:frame>', p, flags=re.S).group(0)).split())
          for p in pagine]


def trova(testo):
    return next(i for i, p in enumerate(pagine) if testo in p)


def sostituisci(i, vecchio, nuovo):
    assert vecchio in pagine[i], (i, vecchio[:60])
    pagine[i] = pagine[i].replace(vecchio, nuovo)


def senza_commenti(i):
    pagine[i] = re.sub(r"<officeooo:annotation.*?</officeooo:annotation>", "", pagine[i], flags=re.S)


def aggiungi_punto(i, testo, corpo=20):
    """Un punto in fondo all'elenco principale della slide, animato all'ultimo clic."""
    ids = []
    voce = elenco([testo], corpo, ids)[len('<text:list text:style-name="L2">'):-len("</text:list>")]
    p = pagine[i]
    fine = p.index("</text:list></draw:text-box>")
    pagine[i] = p[:fine] + voce + p[fine:]
    passo = animazione(ids)
    passo = passo[passo.index("<anim:seq"):]
    passo = passo[passo.index(">") + 1:passo.index("</anim:seq>")]
    pagine[i] = pagine[i].replace("</anim:seq>", passo + "</anim:seq>", 1)


# slide 3 e 4: una sola raccolta, dal catalogo NewsGuard 2026
i = trova("13-18 ottobre: 468 testate; 25 ottobre: altre 825")
sostituisci(i, "13-18 ottobre: 468 testate; 25 ottobre: altre 825", "Ottobre 2026: tutte le 1.293 testate NewsGuard 2026 solo politiche con un voto")
sostituisci(i, "Due campagne, ottobre 2026", "Una sola raccolta, ottobre 2026")
i = trova("Le tre fasi di scraping con dati comulati")
sostituisci(i, "Le tre fasi di scraping con dati comulati", "Un'unica raccolta dal catalogo NewsGuard 2026")
senza_commenti(i)

# slide 12: il titolo non entra nei modelli
i = trova("Il risultato")
sostituisci(i, 'svg:width="25.199cm" svg:height="2cm" svg:x="1.471cm" svg:y="3.4cm"', 'svg:width="25.199cm" svg:height="2.8cm" svg:x="1.471cm" svg:y="3.4cm"')
sostituisci(i, 'svg:width="25cm" svg:height="8cm" svg:x="1.5cm" svg:y="5.4cm"', 'svg:width="23.44cm" svg:height="7.5cm" svg:x="2.28cm" svg:y="6.4cm"')
aggiungi_punto(i, "Anche i **titoli** sono anonimizzati, ma i modelli leggono **solo il testo**: il titolo non entra né in addestramento né in test")
senza_commenti(i)

# slide 14: la lunghezza degli articoli, con i numeri veri
i = trova("Gestione lunghezza articoli")
vecchio = re.search(r'<draw:frame presentation:style-name="pr2".*?</draw:frame>', pagine[i], flags=re.S).group(0)
anim_vecchia = re.search(r"<anim:par .*</anim:par>", pagine[i], flags=re.S)
ids = []
nuovo_corpo = (f'<draw:frame presentation:style-name="pr2" draw:layer="layout" svg:width="25.199cm" svg:height="5.6cm" '
               f'svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline" presentation:user-transformed="true"><draw:text-box>'
               + elenco(["Articoli tra 100 e 1.500 parole: circa **1,33 token per parola**, quindi ~2.000 token",
                         ("Sopra i 2.048 token: solo **56 articoli su 25.524** (0,2%), fino a 2.635", ["con il limite di 2.048 sarebbero stati tagliati"]),
                         "Noi usiamo **4.096** per tutti: **nessun articolo troncato**",
                         "Con 512 token (DeBERTa) sarebbe tagliato il **72%** degli articoli"], 20, ids)
               + "</draw:text-box></draw:frame>")
pagine[i] = pagine[i].replace(vecchio, nuovo_corpo)
if anim_vecchia:
    pagine[i] = pagine[i].replace(anim_vecchia.group(0), "")
pagine[i] = pagine[i].replace("</draw:page>", figura(FIG / "codice" / "token.png", 9.6, basso=12.0) + animazione(ids) + "</draw:page>")
senza_commenti(i)

# ------------------------------------------------------------------------------------------------ slide nuove
proprietari = pagina("Pulizia", "I proprietari: decisi solo qui", [
    "Allo scraping non guardiamo i proprietari: **prima prendiamo i dati, poi decidiamo cosa togliere**",
    ("Tra le 1.293 testate scaricate: **589 gruppi** di proprietà", ["il più grande è una rete di 186 siti con lo stesso Parent Domain"]),
    "Dopo la pulizia restano **287 testate in 226 gruppi**: se ne tolgono 61",
    "Nel dataset finale **nessuna testata sorella**: 226 testate, 226 gruppi",
], corpo=24)

split_perche = "SPLIT_PERCHE"      # si completa con i numeri del controllo (vedi sotto)

modelli = [
    con_grafico("TF-IDF", "Contare le parole, pesare quelle rare", [
        "Ogni articolo diventa un vettore: un peso per ogni parola e coppia di parole (**100.000** in tutto)",
        "Peso alto = parola frequente nell'articolo ma **rara negli altri**",
        "Vocabolario e pesi calcolati **solo sugli articoli di training**",
    ], M / "tfidf_esempio.png"),
    con_codice("TF-IDF", "Classificatore e regressore: cambia solo il modello sopra", [
        "**Classificatore**: regressione logistica, 5 uscite → la probabilità di ogni livello (classi bilanciate)",
        "**Regressore**: ridge, una uscita → il voto 0-100 (minimizza l'errore al quadrato)",
    ], ["tfidf_matrici", "modelli_lineari"]),
    con_grafico("TF-IDF", "Che cosa guarda: le parole più pesanti del regressore", [
        "«outlet» (il segnaposto [OUTLET]) **alza** il voto: chi si cita («said in a statement to [OUTLET]») fa cronaca propria",
        "Frasi fisse, «advertisement», temi come Tehran o Kiev, link a Gettr e Gab lo **abbassano**",
    ], M / "tfidf_parole.png"),
    con_codice("Qwen3-Embedding-8B", "Un modello di embedding, congelato", [
        "8 miliardi di parametri, fatto per trasformare un testo in un vettore (**4.096 numeri**)",
        "Legge un'istruzione + l'articolo intero; il vettore è l'**ultimo token**",
        "**Non lo addestriamo**: sopra, la stessa logistica / ridge del TF-IDF",
    ], ["qwen"]),
    con_codice("Mistral-Small-24B", "Un modello linguistico grande, congelato", [
        "24 miliardi di parametri, versione **Base**; caricato a **4 bit** per entrare nella GPU (32 GB)",
        "Vettore = **media** degli stati interni di tutti i token, strato 27 di 40 (5.120 numeri)",
        "Anche qui **non lo addestriamo**: sopra, logistica o ridge",
    ], ["mistral"]),
    con_codice("ModernBERT", "Un encoder piccolo, addestrato tutto", [
        "149 milioni di parametri, legge fino a 8.192 token; lo **addestriamo tutto** sui nostri articoli",
        "**Classificatore**: 5 uscite, entropia incrociata con classi bilanciate (label smoothing 0,1)",
        "**Regressore**: 1 uscita, errore al quadrato sul voto / 100",
    ], ["modernbert"]),
    pagina("Classificatore e regressore", "Che cosa cambia, per tutti i modelli", [
        ("**Classificatore**: impara il **livello** (l1-l5)", ["dà 5 probabilità per articolo", "la testata = il livello più votato dai suoi articoli"]),
        ("**Regressore**: impara il **voto esatto** (0-100)", ["dà un numero per articolo", "la testata = la media dei voti dei suoi articoli"]),
        "Stessi articoli, stesso split, stessi lettori del testo",
    ]),
]

addestramento = [
    con_codice("Addestramento", "Cinque giri di addestramento e test", [
        "In ogni giro: 4 fold per addestrare, 1 per il test",
        "Ogni testata è prevista **una volta sola**, da un modello che non l'ha mai vista",
        "Lo split: le testate in ordine di voto, a blocchi di 5 (seme fisso)",
    ], ["split"]),
    con_codice("Modelli lineari", "La penalità si sceglie dentro l'addestramento", [
        "3 fold interni sulle sole testate di training, divisi per voto",
        "Ogni penalità della griglia su ogni fold interno: si tiene la migliore (F1 o Spearman, per testata)",
        "Poi il modello si riaddestra su tutto il training e prevede il test",
    ], ["fold_lineare"]),
    con_codice("Modelli lineari", "Le griglie e le penalità scelte", [
        "TF-IDF regressore: 0,3 · 1 · 0,003 · 0,03 · 0,3; Mistral regressore: 100.000 in 3 fold su 5",
        "[[Limite]]: in alcuni fold la scelta cade sul **bordo** della griglia: andrebbe allargata",
    ], ["griglie"]),
    con_codice("ModernBERT", "5 epoche, nessuna validazione", [
        "Learning rate 3·10⁻⁵, batch 4 × 8 accumulati = **32**; warm-up 6%, poi decrescita lineare",
        "Si tiene l'ultima epoca; gradiente tagliato a 1",
        ("Regressore: a fine addestramento sbaglia di ~**4 punti** sul training, di ~**18** sul test", ["→ **impara le testate a memoria**"]),
    ], ["addestramento_bert"]),
    con_codice("Riproducibilità", "Ogni cosa ha un seme", [
        "Seme 42 per tutto; lo split ha seme 1",
        "random_state nei modelli lineari, algoritmi deterministici sulla GPU",
        "Rieseguendo i passi 1-5 i file sono **identici bit per bit**",
        "Salviamo i logit / voti di **ogni articolo di ogni fold** e il modello di ogni fold",
    ], ["riproducibile"]),
]

valutazione = [
    con_codice("Valutazione", "Dagli articoli alla testata", [
        "Classificatore: il **livello più votato** dai suoi articoli (a parità il più basso)",
        "Regressore: la **media** dei voti dei suoi articoli",
        "Intervalli al 95%: **bootstrap sulle testate** (2.000 volte)",
    ], ["da_articoli_a_testata"]),
    con_grafico("Classificatori", "F1 macro sui 5 livelli", ["Senza testo (sempre la classe più frequente): **0,10**"], V / "cls_f1.png"),
    con_grafico("Classificatori", "Le altre misure", [], V / "cls_altre.png"),
    con_grafico("Classificatori", "Dove sbagliano", ["Quasi sempre tra **livelli vicini**: tolerant ±1 fino a 0,83"], V / "cls_confusione.png"),
    con_grafico("Classificatori", "F1 di ogni livello", ["**l3** è il più difficile (F1 0,16-0,24): è una fascia stretta, 60-74"], V / "cls_per_livello.png"),
    con_grafico("Regressori", "Voto vero e voto previsto", ["Le previsioni sono **schiacciate verso il centro**: i modelli non osano gli estremi"], V / "reg_dispersione.png"),
    con_grafico("Regressori", "Errore in punti e ordine delle testate", ["Senza testo (media del training): **22,6 punti**"], V / "reg_mae_spearman.png"),
    con_grafico("Regressori", "Errore per livello vero", ["Gli estremi (l1, l5) sono i più difficili: effetto dello schiacciamento"], V / "reg_per_livello.png"),
    con_codice("Dal voto ai livelli", "Le quote del training", [
        "In ogni fold le testate di test si ordinano per voto previsto",
        "Si tagliano con le **proporzioni dei livelli nel training** dello stesso fold",
    ], ["quote"]),
    con_grafico("Dal voto ai livelli", "Quattro modi a confronto", [
        "Confini NewsGuard fissi: crolla (voti schiacciati); **quote del training**: il migliore",
    ], V / "reg_livelli.png"),
    con_grafico("Classificatore o regressore?", "Stessi modelli, stesse testate", [
        "Sui livelli il regressore con le quote è **pari o meglio**; sul voto si equivalgono",
    ], V / "confronto_cls_reg.png"),
    con_grafico("I modelli contro il TF-IDF", "Differenza sulle stesse testate", [
        "Mistral e Qwen battono il TF-IDF sull'errore del regressore (**−1,9 e −1,6 punti**); ModernBERT no",
    ], V / "contro_tfidf.png"),
    con_grafico("Il DNA", "Quanti articoli servono", [
        "Mistral: con **1** articolo Spearman 0,63, con **5** 0,71, con tutti 0,73",
    ], V / "dna.png"),
    pagina("Anonimizzazione", "Cambia qualcosa?", [
        "Con e senza anonimizzazione i numeri **quasi non cambiano**",
        "Mistral, errore in punti: 14,7 senza (tentativo 18) → **14,8** con",
        "→ i modelli leggono **il testo**, non il nome della testata",
    ]),
    pagina("Media Bias/Fact Check", "Un secondo esperto", [
        ("MBFC valuta il **Factual Reporting** di una testata", ["Very Low, Low, Mixed, Mostly Factual, High, Very High → 0-4"]),
        "Lo confrontiamo con il **voto NewsGuard** (Spearman) e con i suoi livelli",
        "Testate in comune (stesso dominio): **170 su 226**",
        "MBFC non entra mai nell'addestramento: serve solo come **riferimento**",
    ]),
    con_grafico("NewsGuard e MBFC", "Quanto concordano gli esperti", [
        "Spearman **0,71**: due esperti non concordano del tutto",
    ], V / "mbfc_incrocio.png", largo_max=16),
    con_grafico("Il modello come annotatore", "Accordo sulle 170 testate in comune", [
        "Mistral con NewsGuard: **0,73**, come i due esperti tra loro (**0,71**)",
        "Con MBFC un po' meno (0,63): il modello ha imparato NewsGuard",
    ], V / "mbfc_accordo.png"),
    pagina("In breve", "", [
        "Dal **solo testo** anonimo si prevede l'affidabilità di testate mai viste",
        "I modelli grandi congelati sono i migliori; il TF-IDF segue; ModernBERT impara a memoria",
        "Il **regressore** è pari o meglio del classificatore",
        "Mistral concorda con NewsGuard **quanto MBFC**: un annotatore al livello degli esperti",
    ]),
]

# ------------------------------------------------------------------------------------------------ il controllo dello split (dal server)
import pandas as pd   # noqa: E402
ctrl = pd.read_csv(sys.argv[3]) if len(sys.argv) > 3 else None
if ctrl is not None:
    piv = ctrl.set_index(["model", "split"])
    righe = []
    for m in ["TF-IDF", "Qwen3-8B", "Mistral-24B"]:
        a, b = piv.loc[(m, "by score (used)")], piv.loc[(m, "by level (attempt 16)")]
        righe.append(f"{m}: F1 {a['cls F1']:.2f} vs {b['cls F1']:.2f} · MAE {a['reg MAE']:.1f} vs {b['reg MAE']:.1f} · Spearman {a['reg Spearman']:.2f} vs {b['reg Spearman']:.2f}".replace(".", ","))
    split_perche = pagina("Lo split", "Perché per voto, e quanto conta", [
        ("Il regressore impara il **voto esatto**: ogni fold deve coprire **tutta la scala**", ["con i fold per livello, dentro un livello il voto era lasciato al caso"]),
        ("Controllo: stessi modelli con lo **split per livello** del tentativo 16", [f"((per voto vs per livello))  {r}" for r in righe]),
        "→ differenze **piccole** (MAE ±0,3 punti): lo split per voto aiuta un po' il regressore, le conclusioni non cambiano",
    ], corpo=20)

# ------------------------------------------------------------------------------------------------ montaggio
def dopo(testo_titolo):
    return next(i for i, t in enumerate(titoli) if testo_titolo in t)


assert "Passo per passo" in pagine[dopo("Pulizia") + 1]      # la seconda slide «Pulizia» è quella del passo per passo
nuove = {dopo("Pulizia") + 1: [proprietari]}
risolte = [dopo("Pulizia") + 1, dopo("Scelta dei modelli"), dopo("Addestramento"), dopo("Valutazione")]
if split_perche != "SPLIT_PERCHE":
    nuove[dopo("split")] = [split_perche]
    risolte.append(dopo("split"))
nuove[dopo("Scelta dei modelli")] = modelli
nuove[dopo("Addestramento")] = addestramento
nuove[dopo("Valutazione")] = valutazione
for i in risolte:
    senza_commenti(i)
finale = []
for i, p in enumerate(pagine):
    finale.append(p)
    finale += nuove.get(i, [])
corpo = ""
for n, p in enumerate(finale, 1):
    corpo += p.replace('draw:name="NOME"', f'draw:name="page{n}"', 1) if 'draw:name="NOME"' in p else p
inizio, fine = c.find("<draw:page "), c.rfind("</draw:page>") + len("</draw:page>")
c = c[:inizio] + corpo + c[fine:]
for nome in immagini:
    manifest = manifest.replace("</manifest:manifest>", f' <manifest:file-entry manifest:full-path="{nome}" manifest:media-type="image/png"/>\n</manifest:manifest>')
with zipfile.ZipFile(OUT, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
    for info in z.infolist():
        if info.filename == "mimetype":
            continue
        dati = {"content.xml": c.encode(), "META-INF/manifest.xml": manifest.encode()}.get(info.filename) or z.read(info.filename)
        zo.writestr(info, dati, compress_type=zipfile.ZIP_DEFLATED)
    for nome, percorso in immagini.items():
        zo.write(percorso, arcname=nome)
print(len(finale), "slide (erano 17),", contatore[0], "punti nuovi animati ->", OUT)
