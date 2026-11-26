"""Costruisce FinalState.odp nello stile di John (dal suo Regressione.odp: titolo nero con una parte rossa, sottotitolo
verde, elenchi brevi, l'indice ripetuto con la sezione corrente in rosso). Ogni punto compare a un clic. Le sezioni si
aggiungono una alla volta. Prima i grafici:  python figure/genera_grafici.py;  poi:  python build_final.py"""
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

QUI = Path(__file__).resolve().parent
MODELLO = QUI.parent / "Criteri e regressione" / "Regressione.odp"
OUT = QUI / "FinalState.odp"
SEZIONI = ["Scraping", "Pulizia", "Anonimizzazione", "Modelli", "Valutazione"]

z = zipfile.ZipFile(MODELLO)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()
c = c.replace("</office:automatic-styles>", """<style:style style:name="X20" style:family="paragraph"><style:text-properties fo:font-size="20pt" style:font-size-asian="20pt" style:font-size-complex="20pt"/></style:style>
<style:style style:name="X24" style:family="paragraph"><style:text-properties fo:font-size="24pt" style:font-size-asian="24pt" style:font-size-complex="24pt"/></style:style>
<style:style style:name="Xb" style:family="text"><style:text-properties fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"/></style:style>
<style:style style:name="Xr" style:family="text"><style:text-properties fo:color="#c9211e" loext:opacity="100%" fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"/></style:style>
<style:style style:name="Xs" style:family="text"><style:text-properties fo:color="#7f7f7f" loext:opacity="100%"/></style:style>
</office:automatic-styles>""", 1)

contatore = [0]
immagini = {}


def runs(testo):
    """**grassetto**, [[rosso]], ((grigio)) → span."""
    out, pos = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|\[\[(.+?)\]\]|\(\((.+?)\)\)", testo):
        out.append(escape(testo[pos:m.start()]))
        stile, t = [(s, g) for s, g in zip(["Xb", "Xr", "Xs"], m.groups()) if g is not None][0]
        out.append(f'<text:span text:style-name="{stile}">{escape(t)}</text:span>')
        pos = m.end()
    return "".join(out) + escape(testo[pos:])


def elenco(voci, p, ids, stile_lista="L2"):
    """L'elenco puntato; ogni paragrafo riceve un id (per l'animazione) che si aggiunge a `ids`, nell'ordine."""
    xml = f'<text:list text:style-name="{stile_lista}">' if stile_lista else "<text:list>"
    for v in voci:
        testo, figli = (v, []) if isinstance(v, str) else v
        contatore[0] += 1
        pid = f"punto{contatore[0]}"
        ids.append(pid)
        xml += f'<text:list-item><text:p xml:id="{pid}" text:style-name="{p}">{runs(testo)}</text:p>'
        xml += elenco(figli, p, ids, None) if figli else ""
        xml += "</text:list-item>"
    return xml + "</text:list>"


def animazione(ids):
    """Ogni paragrafo compare a un clic, nell'ordine (effetto «Comparsa»), nello stesso formato che scrive LibreOffice:
    senza la durata indefinita della sequenza principale i clic non scattano."""
    passi = "".join(
        f'<anim:par smil:begin="next" smil:fill="hold"><anim:par smil:begin="0s" smil:fill="hold"><anim:par smil:begin="0s" '
        f'smil:fill="hold" presentation:node-type="on-click" presentation:preset-class="entrance" '
        f'presentation:preset-id="ooo-entrance-appear"><anim:set smil:begin="0s" smil:dur="0.001s" smil:fill="hold" '
        f'smil:targetElement="{i}" smil:attributeName="visibility" smil:to="visible"/></anim:par></anim:par></anim:par>'
        for i in ids)
    return (f'<anim:par smil:dur="indefinite" smil:restart="never" presentation:node-type="timing-root">'
            f'<anim:seq smil:dur="indefinite" presentation:node-type="main-sequence">{passi}</anim:seq></anim:par>') if ids else ""


def figura(nome, y, larghezza=25.0):
    """Un grafico della cartella figure/, centrato, in alto a `y` cm."""
    w, h = Image.open(QUI / "figure" / nome).size
    alto = min(larghezza * h / w, 15.4 - y)
    largo = alto * w / h
    immagini[nome] = f"Pictures/{nome}"
    return (f'<draw:frame draw:style-name="gr2" draw:text-style-name="P2" draw:layer="layout" svg:width="{largo:.2f}cm" '
            f'svg:height="{alto:.2f}cm" svg:x="{(28 - largo) / 2:.2f}cm" svg:y="{y:.2f}cm"><draw:image '
            f'xlink:href="Pictures/{nome}" xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad" '
            f'draw:mime-type="image/png"><text:p/></draw:image></draw:frame>')


def pagina(nero, rosso, verde, voci, p="X24", grafico=None, y_grafico=None):
    """Come la slide 5 di John: titolo (nero + rosso), sottotitolo verde, elenco; eventualmente un grafico sotto."""
    ids = []
    alto = 11.6 if grafico is None else y_grafico - 3.4
    titolo = (f'<draw:frame presentation:style-name="pr1" draw:layer="layout" svg:width="25.199cm" svg:height="2.629cm" '
              f'svg:x="1.4cm" svg:y="0cm" presentation:class="title" presentation:user-transformed="true"><draw:text-box>'
              f'<text:p>{escape(nero)}<text:span text:style-name="T4">{escape(rosso)}</text:span></text:p></draw:text-box></draw:frame>')
    sotto = (f'<draw:frame presentation:style-name="pr4" draw:text-style-name="P5" draw:layer="layout" svg:width="25.199cm" '
             f'svg:height="1.102cm" svg:x="1.471cm" svg:y="2.073cm" presentation:class="title" presentation:user-transformed="true">'
             f'<draw:text-box><text:p text:style-name="P4"><text:span text:style-name="T8">{escape(verde)}</text:span></text:p>'
             f'</draw:text-box></draw:frame>')
    corpo = (f'<draw:frame presentation:style-name="pr3" draw:layer="layout" svg:width="25.199cm" svg:height="{alto:.2f}cm" '
             f'svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline" presentation:user-transformed="true">'
             f'<draw:text-box>{elenco(voci, p, ids)}</draw:text-box></draw:frame>')
    img = figura(grafico, y_grafico) if grafico else ""
    return (f'<draw:page draw:name="NOME" draw:style-name="dp1" draw:master-page-name="Default" '
            f'presentation:presentation-page-layout-name="AL2T1">{titolo}{corpo}{img}{sotto}{animazione(ids)}NOTE</draw:page>')


def indice(attiva):
    """La slide indice di John: elenco numerato, la sezione corrente in rosso grassetto."""
    voci = ""
    for i, s in enumerate(SEZIONI):
        voci += f"<text:list-item><text:p>{f'<text:span text:style-name=\"T3\">{s}</text:span>' if i == attiva else s}</text:p></text:list-item>"
    return (f'<draw:page draw:name="NOME" draw:style-name="dp1" draw:master-page-name="Default" '
            f'presentation:presentation-page-layout-name="AL2T1"><draw:frame presentation:style-name="pr1" draw:layer="layout" '
            f'svg:width="25.199cm" svg:height="2.629cm" svg:x="1.4cm" svg:y="0.628cm" presentation:class="title" '
            f'presentation:user-transformed="true"><draw:text-box><text:p>Dal web al giudizio</text:p></draw:text-box>'
            f'</draw:frame><draw:frame presentation:style-name="pr3" draw:layer="layout" svg:width="25.199cm" svg:height="9.134cm" '
            f'svg:x="1.4cm" svg:y="3.685cm" presentation:class="outline" presentation:user-transformed="true"><draw:text-box>'
            f'<text:list text:style-name="L3">{voci}</text:list></draw:text-box></draw:frame>NOTE</draw:page>')


def note(n):
    return (f'<presentation:notes draw:style-name="dp2"><draw:page-thumbnail draw:style-name="gr1" draw:layer="layout" '
            f'svg:width="18.624cm" svg:height="10.476cm" svg:x="1.482cm" svg:y="2.123cm" draw:page-number="{n}" '
            f'presentation:class="page"/><draw:frame presentation:style-name="pr2" draw:text-style-name="P1" '
            f'draw:layer="layout" svg:width="17.271cm" svg:height="12.572cm" svg:x="2.159cm" svg:y="13.271cm" '
            f'presentation:class="notes" presentation:placeholder="true"><draw:text-box/></draw:frame></presentation:notes>')


copertina = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)[0] \
    .replace("Analisi Regressione", "Quanto è affidabile una testata? Chiedilo ai suoi articoli")

# ------------------------------------------------------------------------------------------------ introduzione
introduzione = [
    pagina("La ", "domanda", "Il solo testo basta a dire quanto è affidabile una testata?", [
        "**NewsGuard** valuta le testate: voto 0-100, 5 livelli (l1 < 40 ... l5 = 100), affidabile se ≥ 60",
        "Lo prevediamo **solo dagli articoli**",
        "Su testate **mai viste** in addestramento",
    ]),
    pagina("Quali riguardano il ", "testo", "In rosso: quelli che si vedono in un articolo", [
        ("**Credibility**", ["[[Does not publish false or egregiously misleading content at a significant level]]",
                             "[[Gathers and presents information responsibly]]",
                             "((Has effective practices for correcting errors))",
                             "[[Handles the difference between news and opinion responsibly]]",
                             "[[Avoids deceptive headlines]]"]),
        ("**Transparency**", ["((Website discloses ownership and financing))",
                              "((Clearly labels advertising))",
                              "((Reveals who's in charge, including any possible conflicts of interest))",
                              "((The site provides names of content creators, along with either contact or biographical information))"]),
    ], p="X20"),
    pagina("Gli articoli come ", "DNA", "Una testata è i suoi articoli", [
        "Ogni articolo porta un pezzo del **DNA** della testata",
        "Un articolo da solo: un campione rumoroso",
        "Più articoli: un giudizio sulla testata (voto o media)",
        "Quanti articoli servono? ((lo misuriamo))",
    ]),
    pagina("I modelli come ", "annotatori", "Un terzo annotatore accanto agli esperti", [
        "Il modello legge gli articoli e dà il suo giudizio",
        "Due valutatori esperti, NewsGuard e Media Bias/Fact Check, concordano a **Spearman 0,70** ((170 testate))",
        "→ è questo il riferimento per un modello, non 1",
    ]),
]

# ------------------------------------------------------------------------------------------------ 1. scraping
scraping = [
    indice(0),
    pagina("", "Scraping", "Due campagne, ottobre 2026", [
        "**13-18 ottobre**: 468 testate; **25 ottobre**: altre 825",
        "Testate NewsGuard con il **solo argomento politico**, in **inglese**, con un voto",
        ("Prima di scaricare, da NewsGuard: **gruppi di proprietà** (stesso Owner o Parent Domain)", [
            "1.293 testate in 589 gruppi; il più grande è una rete di 186 siti"]),
        "Fino a **200 articoli** per testata: feed, sitemap, homepage, dai più recenti",
        "Testo, titolo, data, autore con **trafilatura**; si salva l'**HTML grezzo**",
        "Rispetto di robots.txt, una richiesta al secondo per sito",
    ]),
    pagina("Il catalogo ", "NewsGuard 2026", "Il punto di partenza: le testate per voto e per livello", [
        "Nel catalogo un terzo delle testate è l1",
        "Le testate solo politiche sono più bilanciate, il dataset finale ancora di più",
    ], p="X20", grafico="newsguard.png", y_grafico=5.6),
    pagina("", "Scraping", "Che cosa abbiamo ottenuto", [
        "**1.293** testate tentate, **75.647** articoli sopra le 100 parole",
        ("Esito per testata", ["complete (200 articoli): 326", "finiti i link: 456",
                               "homepage irraggiungibile (quasi sempre bloccata): 511"]),
        "**674** testate con almeno un articolo, **86.225** articoli salvati → la pulizia",
    ]),
    pagina("", "Gli articoli scaricati", "Quanti per testata, quanti per livello", [
        "Metà delle testate con articoli arriva a 200; molte ne danno pochi",
    ], p="X20", grafico="articoli_scaricati.png", y_grafico=4.8),
    pagina("Prima: ", "che cosa non ha funzionato", "Etichette vecchie", [
        "Primi tentativi: etichette NewsGuard esportate ad **agosto 2022**",
        "→ circa **4 anni più vecchie** degli articoli",
        "Ora: **NewsGuard 2026** (ottobre), lo stesso periodo dello scraping",
    ]),
]

# ------------------------------------------------------------------------------------------------ 2. pulizia
pulizia = [
    indice(1),
    pagina("", "Pulizia", "Regole semplici, fissate prima dei risultati", [
        "Testo riestratto dall'HTML con trafilatura, modalità **precisione**",
        "Inglese, 100-1.500 parole, pubblicati dal 2025",
        ("Niente **doppioni** né quasi doppioni, tra tutte le testate", [
            "quasi doppioni: due articoli che hanno in comune almeno metà dei gruppi di 5 parole consecutive (MinHash, Jaccard ≥ 0,5)"]),
        "Almeno 10 articoli per testata",
        "**Una testata per gruppo di proprietà**: quella con più articoli",
    ], p="X20"),
    pagina("", "Pulizia", "Passo per passo", [
        "Articoli: i tagli più grandi sono lingua, lunghezza e data",
        "Testate: il taglio più grande sono i **doppioni** (da 531 a 361: siti che ripubblicano)",
        "Infine una testata per gruppo di proprietà, quella con più articoli: da 287 a **226**",
    ], p="X20", grafico="pulizia.png", y_grafico=7.0),
    pagina("", "Doppioni", "Tolti dal dataset, non solo separati tra i fold", [
        ("**Stesso articolo** (testo identico): si tiene la copia più vecchia", [
            "5.433 articoli tolti; 170 testate spariscono: erano copie di altre"]),
        ("**Articolo simile** (quasi doppione, Jaccard ≥ 0,5): si tiene la copia più vecchia", [
            "8.403 coppie: 7.120 nella stessa testata, 1.283 tra testate diverse"]),
        ("**Stessa proprietà**: una sola testata per gruppo", [
            "nel dataset **non ci sono testate sorelle**: 226 testate, 226 gruppi diversi"]),
        "I fold sono di testate: una testata è tutta in addestramento o tutta in test",
    ], p="X20"),
    pagina("Il ", "testo", "Pulito con la modalità automatica di trafilatura", [
        "Trafilatura in modalità precisione: tiene il testo dell'articolo, toglie menu, pubblicità, commenti, link",
        "Nessuna regola di pulizia scritta da noi",
    ]),
    pagina("Il ", "dataset finale", "226 testate, 25.524 articoli", [
        "Tutti i livelli rappresentati, l4 il più numeroso",
        "Articoli soprattutto degli ultimi mesi (lo scraping parte dai più recenti)",
    ], p="X20", grafico="dataset_finale.png", y_grafico=5.0),
]

# ------------------------------------------------------------------------------------------------ 3. anonimizzazione
anonimizzazione = [
    indice(2),
    pagina("", "Anonimizzazione", "Perché", [
        "I modelli grandi hanno letto il web: **conoscono le testate**",
        "Se nel testo resta il nome, il modello può riconoscere la testata invece di leggere l'articolo",
        "Togliamo tutto ciò che dice **chi** l'ha pubblicato, niente di ciò che dice **che cosa** racconta",
    ]),
    pagina("", "Anonimizzazione", "Sei regole semplici, nell'ordine", [
        "**Firma** nei titoli (la coda ripetuta: \"... | The New York Sun\") → [OUTLET]",
        "Email → [EMAIL], link → [URL], account @ → [ACCOUNT]",
        "**Autori** della testata → [AUTHOR]",
        "**Nome della testata** dal dominio, attaccato o staccato, con o senza \"the\" → [OUTLET]",
        "Esempio: \"Sign up to receive [OUTLET]'s biggest investigations\"",
    ], p="X20"),
    pagina("", "Anonimizzazione", "I casi difficili, risolti con regole", [
        ("Nomi che sono parole comuni (politics, time, reason)", ["tolti solo con la maiuscola: Politics, TIME"]),
        ("Nomi diversi dal dominio", ["li trova la firma nei titoli: cfr.org → \"Council on Foreign Relations\""]),
        ("Autori", ["solo chi firma per una sola testata: le agenzie restano"]),
        "Nessuna lettura a mano, nessun LLM: **controlli automatici**",
    ], p="X20"),
    pagina("", "Anonimizzazione", "Il risultato", [
        "Il **54,6%** degli articoli cambia almeno una parola",
        "Controllo: **nessun nome** di testata rimasto come parola intera",
    ], p="X20", grafico="anonimizzazione.png", y_grafico=5.4),
]

# ------------------------------------------------------------------------------------------------ 4. modelli: prima della classificazione
modelli = [
    indice(3),
    pagina("Lo ", "split", "Cinque fold di testate, per voto", [
        "Testate in ordine di voto, a blocchi di 5: ognuna in un fold diverso",
        "Ogni fold ha lo stesso voto medio e le stesse testate per livello (±1)",
        "Lo stesso split per classificazione e regressione",
        ("Prima: fold per livello", ["le l1 avevano voto medio 12 in un fold, 31 in un altro"]),
    ], p="X20", grafico="split.png", y_grafico=8.2),
    pagina("Il testo ", "intero", "Nessun articolo troncato", [
        "Articoli tra 100 e 1.500 parole",
        "Il più lungo: **3.000 token** (mediana 760); limite **4.096** per tutti i modelli",
        "Prima: con 512 o 2.048 token una parte degli articoli lunghi veniva tagliata",
    ]),
]

diapositive = introduzione + scraping + pulizia + anonimizzazione + modelli
corpo = ""
for n, s in enumerate(diapositive, 1):
    s = re.sub(r'draw:name="[^"]*"', f'draw:name="page{n}"', s, count=1)
    s = s.replace("NOTE", note(n))
    corpo += re.sub(r'draw:page-number="\d+"', f'draw:page-number="{n}"', s)
inizio, fine = c.find("<draw:page "), c.rfind("</draw:page>") + len("</draw:page>")
c = c[:inizio] + corpo + c[fine:]
for nome, dest in immagini.items():
    manifest = manifest.replace("</manifest:manifest>",
                                f' <manifest:file-entry manifest:full-path="{dest}" manifest:media-type="image/png"/>\n</manifest:manifest>')

with zipfile.ZipFile(OUT, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
    for info in z.infolist():
        if info.filename == "mimetype":
            continue
        dati = {"content.xml": c.encode(), "META-INF/manifest.xml": manifest.encode()}.get(info.filename) or z.read(info.filename)
        zo.writestr(info, dati, compress_type=zipfile.ZIP_DEFLATED)
    for nome, dest in immagini.items():
        zo.write(QUI / "figure" / nome, arcname=dest)
print(len(diapositive), "diapositive,", contatore[0], "punti animati ->", OUT)
