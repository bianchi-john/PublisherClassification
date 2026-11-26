"""Gli strumenti per modificare FinalState.odp nello stile di John: slide nuove (titolo rosso, sottotitolo verde, testo
24 o 20 pt, un punto per clic) e piccole modifiche alle slide che esistono già (togliere o riscrivere un punto, cambiare
un'immagine, togliere i commenti), senza toccare il resto. Le slide nuove usano stili propri «Z…» con gli stessi
caratteri e colori di John (aggiunti da `aggiungi_stili`), più i suoi stili di pagina ed elenco."""
import html
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

QUI = Path(__file__).resolve().parent
FIG = QUI / "figure"
# Stili propri (nomi fissi «Z…»), aggiunti al file da aggiungi_stili: LibreOffice rinumera i suoi stili automatici
# (T3, T13, gr2, ...) a ogni salvataggio, quindi i nomi di John cambiano significato da una versione all'altra.
STILI = {24: {"": "ZT24", "b": "ZT24b", "r": "ZT24r", "g": "ZT20g"}, 20: {"": "ZT20", "b": "ZT20b", "r": "ZT20r", "g": "ZT20g"}}
TITOLO, SOTTOTITOLO, IMMAGINE = "ZTtitolo", "ZTsottotitolo", "ZGimmagine"
_DEFINIZIONI = {"ZT24": ("24pt", "normal", None), "ZT24b": ("24pt", "bold", None), "ZT24r": ("24pt", "bold", "#c9211e"),
                "ZT20": ("20pt", "normal", None), "ZT20b": ("20pt", "bold", None), "ZT20r": ("20pt", "bold", "#c9211e"),
                "ZT20g": ("20pt", "normal", "#7f7f7f"), "ZTtitolo": (None, None, "#c9211e"), "ZTsottotitolo": ("28pt", None, "#00a933")}
immagini = {}            # immagini nuove da mettere nel file: nome dentro l'odp -> file
contatore = [0]          # per gli xml:id dei paragrafi nuovi
PREFISSO = ["zp"]        # prefisso degli xml:id nuovi (diverso a ogni revisione, per non scontrarsi)


def aggiungi_stili(c):
    """Aggiunge a content.xml gli stili «Z…» delle slide nuove (se non ci sono già) e controlla gli stili di pagina e di
    elenco che le slide nuove prendono dal file di John (pr1 titolo, pr2 elenco, pr4 sottotitolo, L2, dp1)."""
    for st in ["L2", "pr1", "pr2", "pr4", "dp1"]:
        assert f'style:name="{st}"' in c, st
    nuovi = ""
    for nome, (corpo, peso, colore) in _DEFINIZIONI.items():
        if f'style:name="{nome}"' in c:
            continue
        p = "".join([f' fo:font-size="{corpo}" style:font-size-asian="{corpo}" style:font-size-complex="{corpo}"' if corpo else "",
                     f' fo:font-weight="{peso}" style:font-weight-asian="{peso}" style:font-weight-complex="{peso}"' if peso else "",
                     f' fo:color="{colore}"' if colore else ""])
        nuovi += f'<style:style style:name="{nome}" style:family="text"><style:text-properties{p}/></style:style>'
    if f'style:name="{IMMAGINE}"' not in c:
        nuovi += (f'<style:style style:name="{IMMAGINE}" style:family="graphic" style:parent-style-name="Object_20_with_20_no_20_fill_20_and_20_no_20_line">'
                  f'<style:graphic-properties draw:stroke="none" draw:fill="none" fo:clip="rect(0cm, 0cm, 0cm, 0cm)"/></style:style>')
    return c.replace("</office:automatic-styles>", nuovi + "</office:automatic-styles>", 1)


# ------------------------------------------------------------------------------------------------ slide nuove
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
        pid = f"{PREFISSO[0]}{contatore[0]}"
        ids.append(pid)
        xml += f'<text:list-item><text:p xml:id="{pid}" text:id="{pid}">{runs(testo, corpo)}</text:p>'
        xml += elenco(figli, corpo, ids, False) if figli else ""
        xml += "</text:list-item>"
    return xml + "</text:list>"


def passi(ids):
    """Un clic per paragrafo (ooo-entrance-appear), come li salva LibreOffice."""
    return "".join(
        f'<anim:par smil:begin="next" smil:fill="hold"><anim:par smil:begin="0s" smil:fill="hold"><anim:par smil:begin="0s" '
        f'smil:fill="hold" presentation:node-type="on-click" presentation:preset-class="entrance" '
        f'presentation:preset-id="ooo-entrance-appear"><anim:set smil:begin="0s" smil:dur="0.001s" smil:fill="hold" '
        f'smil:targetElement="{i}" smil:attributeName="visibility" smil:to="visible"/></anim:par></anim:par></anim:par>' for i in ids)


def animazione(ids):
    return (f'<anim:par smil:dur="indefinite" smil:restart="never" presentation:node-type="timing-root"><anim:seq '
            f'smil:dur="indefinite" presentation:node-type="main-sequence">{passi(ids)}</anim:seq></anim:par>') if ids else ""


def nome_immagine(percorso):
    nome = f"Pictures/z_{percorso.parent.name}_{percorso.name}"
    immagini[nome] = percorso
    return nome


def immagine(percorso, x, y, largo=None, alto=None):
    """Un'immagine: data la larghezza o l'altezza (cm), l'altra segue le proporzioni."""
    w, h = Image.open(percorso).size
    if largo is None:
        largo = alto * w / h
    alto = largo * h / w
    return (f'<draw:frame draw:style-name="{IMMAGINE}" draw:layer="layout" svg:width="{largo:.2f}cm" svg:height="{alto:.2f}cm" '
            f'svg:x="{x:.2f}cm" svg:y="{y:.2f}cm"><draw:image xlink:href="{nome_immagine(percorso)}" xlink:type="simple" '
            f'xlink:show="embed" xlink:actuate="onLoad" draw:mime-type="image/png"><text:p/></draw:image></draw:frame>')


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
         f'<text:p><text:span text:style-name="{TITOLO}">{escape(titolo)}</text:span></text:p></draw:text-box></draw:frame>')
    s = (f'<draw:frame presentation:style-name="pr4" draw:text-style-name="P5" draw:layer="layout" svg:width="25.199cm" '
         f'svg:height="1.102cm" svg:x="1.471cm" svg:y="2.073cm" presentation:class="title" presentation:user-transformed="true">'
         f'<draw:text-box><text:p><text:span text:style-name="{SOTTOTITOLO}">{escape(sottotitolo)}</text:span></text:p></draw:text-box>'
         f'</draw:frame>') if sottotitolo else ""
    b = (f'<draw:frame presentation:style-name="pr2" draw:layer="layout" svg:width="25.199cm" svg:height="{alto_testo:.2f}cm" '
         f'svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline" presentation:user-transformed="true"><draw:text-box>'
         f'{elenco(voci, corpo, ids)}</draw:text-box></draw:frame>') if voci else ""
    return (f'<draw:page draw:name="NOME" draw:style-name="dp1" draw:master-page-name="Default" '
            f'presentation:presentation-page-layout-name="AL1T1">{t}{b}{extra}{s}{animazione(ids)}</draw:page>')


def altezza_testo(voci, corpo=20):
    """Altezza (cm) dell'elenco: le righe che vanno a capo contano (circa 76 caratteri per riga a 20 pt, 63 a 24 pt)."""
    per_riga = {20: 76, 24: 63}[corpo]
    alto = 0.0
    for v in voci:
        testo_v, figli = (v, []) if isinstance(v, str) else v
        for t, rientro in [(testo_v, 0)] + [(f, 8) for f in figli]:
            righe = -(-len(re.sub(r"\*\*|\[\[|\]\]|\(\(|\)\)", "", t)) // (per_riga - rientro))
            alto += righe * (0.8 if corpo == 20 else 0.95) + 0.2
    return alto


def con_grafico(titolo, sottotitolo, voci, grafico, corpo=20, y=None, largo_max=25.0):
    """Testo breve in alto, grafico sotto."""
    y = y or (3.6 + altezza_testo(voci, corpo) + 0.3 if voci else 3.4)
    return pagina(titolo, sottotitolo, voci, corpo, alto_testo=max(y - 3.4, 0.8), extra=figura(grafico, y, largo_max))


def con_codice(titolo, sottotitolo, voci, codici, corpo=20, y=None, cm_per_pixel=0.0135):
    """Testo breve in alto, una o più schermate di codice sotto, una sotto l'altra, tutte alla stessa scala (stessa
    grandezza del carattere): la più grande possibile fino a cm_per_pixel, centrate."""
    y = y or (3.6 + altezza_testo(voci, corpo) + 0.3)
    file = [FIG / "codice" / f"{cod}.png" for cod in codici]
    misure = [Image.open(f).size for f in file]
    spazio = 0.25
    scala = min(cm_per_pixel, 25.0 / max(w for w, _ in misure), (15.3 - y - spazio * (len(file) - 1)) / sum(h for _, h in misure))
    extra, cima = "", y
    for f, (w, h) in zip(file, misure):
        extra += immagine(f, (28 - w * scala) / 2, cima, largo=w * scala)
        cima += h * scala + spazio
    return pagina(titolo, sottotitolo, voci, corpo, alto_testo=y - 3.4, extra=extra)


# ------------------------------------------------------------------------------------------------ slide che esistono già
def testo(xml):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", xml)).split())


def titolo_di(p):
    return testo(re.search(r'presentation:class="title".*?</draw:frame>', p, flags=re.S).group(0))


def senza_commenti(p):
    return re.sub(r"<officeooo:annotation.*?</officeooo:annotation>", "", p, flags=re.S)


def _fine(s, inizio, tag):
    """La fine dell'elemento `tag` che comincia in `inizio` (contando gli elementi dello stesso tipo annidati)."""
    livello, pos = 0, inizio
    apre, chiude = re.compile(f"<{tag}[ >/]"), f"</{tag}>"
    while True:
        a, c = apre.search(s, pos), s.find(chiude, pos)
        if a and a.start() < c:
            livello += 1; pos = a.end()
        else:
            livello -= 1; pos = c + len(chiude)
            if livello == 0:
                return pos


def togli_animazioni(p, ids):
    """Toglie i clic che mostrano i paragrafi `ids` (solo i clic che riguardano soltanto quei paragrafi)."""
    for i in ids:
        bersaglio = p.find(f'smil:targetElement="{i}"')
        if bersaglio < 0:
            continue
        inizio = p.rfind('<anim:par smil:begin="next"', 0, bersaglio)
        fine = _fine(p, inizio, "anim:par")
        assert set(re.findall(r'smil:targetElement="([^"]+)"', p[inizio:fine])) <= set(ids), "clic con altri paragrafi"
        p = p[:inizio] + p[fine:]
    return p


def togli_punto(p, contiene):
    """Toglie il punto dell'elenco il cui testo contiene `contiene` (con i suoi sottopunti) e i suoi clic."""
    for m in re.finditer(r"<text:p(?:>| [^>]*?(?<!/)>).*?</text:p>", p, flags=re.S):
        if contiene in testo(m.group(0)):
            inizio = p.rfind("<text:list-item>", 0, m.start())
            fine = _fine(p, inizio, "text:list-item")
            ids = re.findall(r'xml:id="([^"]+)"', p[inizio:fine])
            p = p[:inizio] + p[fine:]
            p = p.replace("<text:list></text:list>", "")
            return togli_animazioni(p, ids)
    raise KeyError(contiene)


def riscrivi(p, vecchio, nuovo, corpo=20):
    """Il paragrafo che contiene `vecchio` prende il testo `nuovo` (stesso id, quindi stesso clic)."""
    for m in re.finditer(r"(<text:p(?:>| [^>]*?(?<!/)>))(.*?)</text:p>", p, flags=re.S):
        if vecchio in testo(m.group(0)):
            return p[:m.start()] + m.group(1) + runs(nuovo, corpo) + "</text:p>" + p[m.end():]
    raise KeyError(vecchio)


def cambia_immagine(p, percorso, basso=15.3):
    """L'immagine della slide diventa `percorso`: stessa x, y e larghezza, altezza dalle proporzioni (se esce dalla slide
    si rimpicciolisce e si centra)."""
    m = re.search(r'<draw:frame draw:style-name="(?:gr\d+|ZGimmagine)"[^>]*svg:width="([\d.]+)cm" svg:height="([\d.]+)cm" '
                  r'svg:x="([\d.]+)cm" svg:y="([\d.]+)cm"[^>]*><draw:image xlink:href="([^"]+)"', p)
    largo, _, x, y, vecchia = m.groups()
    largo, x, y = float(largo), float(x), float(y)
    w, h = Image.open(percorso).size
    if y + largo * h / w > basso:
        nuovo_largo = (basso - y) * w / h
        x, largo = x + (largo - nuovo_largo) / 2, nuovo_largo
    cornice = re.sub(r'svg:width="[\d.]+cm" svg:height="[\d.]+cm" svg:x="[\d.]+cm" svg:y="[\d.]+cm"',
                     f'svg:width="{largo:.2f}cm" svg:height="{largo * h / w:.2f}cm" svg:x="{x:.2f}cm" svg:y="{y:.2f}cm"', m.group(0))
    return p.replace(m.group(0), cornice.replace(vecchia, nome_immagine(percorso)))


# ------------------------------------------------------------------------------------------------ scrittura
def scrivi(origine, uscita, c, manifest, pagine):
    """Rimette le pagine in content.xml (numerandole) e scrive il nuovo .odp con le immagini nuove."""
    corpo = ""
    for n, p in enumerate(pagine, 1):
        corpo += re.sub(r'draw:name="[^"]*"', f'draw:name="page{n}"', p, count=1)
    inizio, fine = c.find("<draw:page "), c.rfind("</draw:page>") + len("</draw:page>")
    c = c[:inizio] + corpo + c[fine:]
    for nome in immagini:
        if f'manifest:full-path="{nome}"' not in manifest:
            manifest = manifest.replace("</manifest:manifest>", f' <manifest:file-entry manifest:full-path="{nome}" manifest:media-type="image/png"/>\n</manifest:manifest>')
    z = zipfile.ZipFile(origine)
    with zipfile.ZipFile(uscita, "w") as zo:
        zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
        for info in z.infolist():
            if info.filename == "mimetype" or info.filename in immagini:
                continue
            dati = {"content.xml": c.encode(), "META-INF/manifest.xml": manifest.encode()}.get(info.filename) or z.read(info.filename)
            zo.writestr(info, dati, compress_type=zipfile.ZIP_DEFLATED)
        for nome, percorso in immagini.items():
            zo.write(percorso, arcname=nome)
