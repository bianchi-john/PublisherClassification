"""Completa Regressione.odp nello stile di John: copia la struttura delle sue slide (titolo nero + parte rossa, sottotitolo
verde, elenchi) e aggiunge le slide mancanti. Scrive un file nuovo, l'originale non si tocca.
Uso: python completa.py <Regressione.odp> <cartella figure> <uscita.odp>"""
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ORIG, FIG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
z = zipfile.ZipFile(ORIG)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()

# stili nuovi: corpo a 20 e 24 pt, grassetto, rosso, verde, corsivo
NUOVI = """<style:style style:name="X20" style:family="paragraph"><style:text-properties fo:font-size="20pt" style:font-size-asian="20pt" style:font-size-complex="20pt"/></style:style>
<style:style style:name="X24" style:family="paragraph"><style:text-properties fo:font-size="24pt" style:font-size-asian="24pt" style:font-size-complex="24pt"/></style:style>
<style:style style:name="Xb" style:family="text"><style:text-properties fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"/></style:style>
<style:style style:name="Xr" style:family="text"><style:text-properties fo:color="#c9211e" loext:opacity="100%" fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"/></style:style>
<style:style style:name="Xg" style:family="text"><style:text-properties fo:color="#00a933" loext:opacity="100%" fo:font-weight="bold" style:font-weight-asian="bold" style:font-weight-complex="bold"/></style:style>
<style:style style:name="Xi" style:family="text"><style:text-properties fo:font-style="italic" style:font-style-asian="italic" style:font-style-complex="italic"/></style:style>
"""
c = c.replace("</office:automatic-styles>", NUOVI + "</office:automatic-styles>", 1)


def runs(testo):
    """**grassetto**, [[rosso]], {{verde}}, __corsivo__ → span."""
    out, pos = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|\[\[(.+?)\]\]|\{\{(.+?)\}\}|__(.+?)__", testo):
        out.append(escape(testo[pos:m.start()]))
        stile, t = [(s, g) for s, g in zip(["Xb", "Xr", "Xg", "Xi"], m.groups()) if g is not None][0]
        out.append(f'<text:span text:style-name="{stile}">{escape(t)}</text:span>')
        pos = m.end()
    out.append(escape(testo[pos:]))
    return "".join(out)


def elenco(voci, p, stile_lista="L2"):
    xml = f'<text:list text:style-name="{stile_lista}">' if stile_lista else "<text:list>"
    for v in voci:
        testo, figli = (v, []) if isinstance(v, str) else v
        xml += f'<text:list-item><text:p text:style-name="{p}">{runs(testo)}</text:p>'
        if figli:
            xml += elenco(figli, p, None)
        xml += "</text:list-item>"
    return xml + "</text:list>"


def note(n):
    return (f'<presentation:notes draw:style-name="dp2"><draw:page-thumbnail draw:style-name="gr1" draw:layer="layout" '
            f'svg:width="18.624cm" svg:height="10.476cm" svg:x="1.482cm" svg:y="2.123cm" draw:page-number="{n}" '
            f'presentation:class="page"/><draw:frame presentation:style-name="pr2" draw:text-style-name="P1" draw:layer="layout" '
            f'svg:width="17.271cm" svg:height="12.572cm" svg:x="2.159cm" svg:y="13.271cm" presentation:class="notes" '
            f'presentation:placeholder="true"><draw:text-box/></draw:frame></presentation:notes>')


immagini = []


def figura(nome, larghezza, x, y):
    from PIL import Image
    w, h = Image.open(FIG / nome).size
    dest = f"Pictures/{Path(nome).stem}.png"
    immagini.append((FIG / nome, dest))
    alt = larghezza * h / w
    return (f'<draw:frame draw:style-name="gr2" draw:text-style-name="P2" draw:layer="layout" svg:width="{larghezza:.2f}cm" '
            f'svg:height="{alt:.2f}cm" svg:x="{x:.2f}cm" svg:y="{y:.2f}cm"><draw:image xlink:href="{dest}" xlink:type="simple" '
            f'xlink:show="embed" xlink:actuate="onLoad" draw:mime-type="image/png"><text:p/></draw:image></draw:frame>')


def pagina(nero, rosso, verde, voci, p="X20", alto_testo=3.4, h_testo=11.5, extra=""):
    """Come la slide 5 di John: titolo in alto (nero + rosso), sottotitolo verde, elenco sotto."""
    titolo = (f'<draw:frame presentation:style-name="pr1" draw:layer="layout" svg:width="25.199cm" svg:height="2.629cm" '
              f'svg:x="1.4cm" svg:y="0cm" presentation:class="title" presentation:user-transformed="true"><draw:text-box>'
              f'<text:p>{escape(nero)}<text:span text:style-name="T4">{escape(rosso)}</text:span></text:p></draw:text-box></draw:frame>')
    sotto = (f'<draw:frame presentation:style-name="pr4" draw:text-style-name="P5" draw:layer="layout" svg:width="25.199cm" '
             f'svg:height="1.102cm" svg:x="1.471cm" svg:y="2.073cm" presentation:class="title" presentation:user-transformed="true">'
             f'<draw:text-box><text:p text:style-name="P4"><text:span text:style-name="T8">{escape(verde)}</text:span></text:p>'
             f'</draw:text-box></draw:frame>')
    corpo = ""
    if voci:
        corpo = (f'<draw:frame presentation:style-name="pr3" draw:layer="layout" svg:width="25.199cm" svg:height="{h_testo}cm" '
                 f'svg:x="1.471cm" svg:y="{alto_testo}cm" presentation:class="outline" presentation:user-transformed="true">'
                 f'<draw:text-box>{elenco(voci, p)}</draw:text-box></draw:frame>')
    return (f'<draw:page draw:name="NOME" draw:style-name="dp1" draw:master-page-name="Default" '
            f'presentation:presentation-page-layout-name="AL2T1">{titolo}{corpo}{extra}{sotto}NOTE</draw:page>')


def indice(attiva):
    voci = ""
    for i, v in enumerate(["Per classi", "Voto degli articoli", "Punteggio esatto"]):
        testo = f'<text:span text:style-name="T3">{v}</text:span>' if i == attiva else v
        voci += f"<text:list-item><text:p>{testo}</text:p></text:list-item>"
    return (f'<draw:page draw:name="NOME" draw:style-name="dp1" draw:master-page-name="Default" '
            f'presentation:presentation-page-layout-name="AL2T1"><draw:frame presentation:style-name="pr1" draw:layer="layout" '
            f'svg:width="25.199cm" svg:height="2.629cm" svg:x="1.4cm" svg:y="0.628cm" presentation:class="title" '
            f'presentation:user-transformed="true"><draw:text-box><text:p>Le tre strategie di regressione</text:p></draw:text-box>'
            f'</draw:frame><draw:frame presentation:style-name="pr3" draw:layer="layout" svg:width="25.199cm" svg:height="9.134cm" '
            f'svg:x="1.4cm" svg:y="3.685cm" presentation:class="outline" presentation:user-transformed="true"><draw:text-box>'
            f'<text:list text:style-name="L3">{voci}</text:list></draw:text-box></draw:frame>NOTE</draw:page>')


# ---------------------------------------------------------------- le pagine di John (prese così come sono)
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(pagine) == 7
p1, p2, p3, p4, p5, p6, p7 = pagine
p2 = p2.replace("(previsto −vero)2", "(previsto − vero)²")

# slide 5: le sue frasi, con il pezzo mancante completato
s5 = pagina("Regressione ", "per classi", "Come si valuta?", [
    ("Il regressore produce un valore 0-100 per i singoli articoli", []),
    ("Per ogni testata si fa la media e si vede in che label cade", ["Quindi solo valutazione per testata"]),
    ("Quando si valuta __reputabili vs. non reputabili__ si prendono le valutazioni per articolo e si fa la media semplicemente",
     ["media ≥ 60 → reputabile: **nessun riadattamento**"]),
    ("Quando si valuta invece per tutte le classi di reputability:",
     [("Si riadattano i valori **con le quote del training**:", [
         "dal **training** solo le **percentuali** delle classi: quante testate l1, l2, …",
         "dal **test** le medie previste, in fila: si taglia la fila con quelle percentuali → **le soglie**"])]),
])

soglie1 = pagina("Regressione ", "per classi", "Da dove vengono le soglie? (1/2)", [
    ("**1) Dal training: quante testate per classe**", [
        "181 testate di training: l1 28, l2 35, l3 23, l4 63, l5 32 (le etichette vere)",
        "cioè 15,5% | 19,3% | 12,7% | 34,8% | 17,7%",
        "quante stanno **in quel livello o sotto**: l1 15,5% · l1+l2 15,5+19,3 = **34,8%** · fino a l3 +12,7 = **47,5%** · fino a l4 +34,8 = **82,3%**"]),
    ("**2) Dal test: le 45 testate in fila, per media prevista**", [
        "36,5  39,4  41,0  [[41,7]]  42,7  45,4  49,0 | 54,1 … 60,3 | 60,6 … 65,2 | 67,8 … 84,6 | 84,9 … 89,7",
        "[[41,7]] è lifezette.com"]),
    "Del training si usano **solo le percentuali**, non i voti",
])

soglie2 = pagina("Regressione ", "per classi", "Da dove vengono le soglie? (2/2)", [
    ("**3) Si taglia la fila con le percentuali del training**", [
        "15,5% di 45 ≈ 7 → le 7 più basse sono **l1**",
        "34,8% di 45 ≈ 16 → dalla 8ª alla 16ª sono **l2**",
        "47,5% di 45 ≈ 21 → dalla 17ª alla 21ª sono **l3**",
        "82,3% di 45 ≈ 37 → dalla 22ª alla 37ª sono **l4**, le ultime 8 sono **l5**"]),
    ("**4) La soglia è un valore tra due testate vicine**", [
        "tra la 7ª (49,0) e l'8ª (54,1) → **53,1**;   tra 60,3 e 60,6 → **60,4**",
        "tra 65,2 e 67,8 → **67,6**;   tra 84,6 e 84,9 → **84,7**"]),
    "Le soglie **cambiano da fold a fold**: dipendono dalle previsioni di quelle 45 testate",
])

esempio1 = pagina("Regressione ", "per classi", "Esempio: lifezette.com", [
    "31 articoli, voto vero **0** (l1, non reputabile)",
    "Voti previsti per gli articoli: 22, 34, 42, 44, 39, 37, 71, 26, …",
    ("Media della testata: **41,7**", []),
    ("Reputabile? 41,7 < 60 → **no** {{✓}}", []),
    ("Classe? Soglie ricavate dalle 45 testate di test: 53,1 | 60,4 | 67,6 | 84,7", ["41,7 è sotto 53,1 → **l1** {{✓}}"]),
    "La distanza 41,7 contro 0 **non si guarda**: conta solo in che classe cade",
], p="X24")

schiaccia = pagina("Perché ", "riadattare?", "Il regressore schiaccia verso il centro", [
    "Voti veri da 0 a 100, medie previste solo da **36 a 111**",
    "Con le soglie di NewsGuard (40 | 60 | 75 | 100) l1 e l5 non escono quasi mai: [[4 testate l1]] (vere 35), [[1 l5]] (vere 39)",
], extra=figura("schiacciamento.png", 23.5, 2.25, 7.6), h_testo=4.2)

stira = pagina("Come si ", "riadatta?", "Come stirare un'immagine, ma per ordine", [
    "1) nessuno   2) min–max = il contrasto di una foto   3) **quote** = equalizzare l'istogramma",
], extra=figura("stiramento.png", 19.5, 4.25, 5.3), h_testo=1.6)

quote = pagina("Come si ", "riadatta?", "Le quote, passo passo", [
    "Ogni classe prende **la stessa quota che ha nel training**: 7, 9, 5, 16, 8 testate (vere: 7, 8, 7, 16, 7)",
    "Conta solo **l'ordine** delle testate, non la distanza in punti",
    "Le etichette di test **non si usano mai**",
], extra=figura("quote_passo_passo.png", 25.0, 1.5, 8.3), h_testo=4.6)

confronto = pagina("Quale ", "riadattamento?", "Sulle 226 testate", [
    ("**5 classi** (classi giuste, F1)", [
        "nessuno: 94 giuste, F1 0,32 → quasi niente l1 e l5",
        "min–max: 69 giuste, F1 0,26 → dipende da una sola testata per estremo",
        "**quote: 95 giuste, F1 0,40** ← quello usato"]),
    ("**Reputabili vs. non reputabili** (accuratezza)", [
        "media ≥ 60, nessun riadattamento: **0,83** (150 previste reputabili, vere 148) ← quello usato",
        "con le quote: 0,81      con min–max: 0,69",
        "60 sta in mezzo, e lì le previsioni si spostano poco: non serve riadattare"]),
])

# slide 7: le sue (titolo con "voto" tutto rosso), riempita
s7 = pagina("Regressione per ", "voto degli articoli", "Come si valuta?", [
    "Stesso regressore, stessi valori 0-100 per articolo",
    ("Ogni **articolo** diventa una classe", ["con le quote **degli articoli** del training (non delle testate)"]),
    ("La testata prende la classe **più votata** dai suoi articoli", ["a parità vince la classe più bassa"]),
    ("__Reputabili vs. non reputabili__: la maggioranza degli articoli è in l3–l5?", []),
    ("Serve per sapere **quanti articoli bastano** (1, 5, 25…)", []),
])

esempio2 = pagina("Regressione per ", "voto degli articoli", "Esempio: lifezette.com", [
    ("Soglie: **stesso calcolo** della 1), ma la fila sono i 5.481 **articoli** di test", [
        "e le percentuali vengono dai 20.043 **articoli** di training: 17,3% | 39,3% | 52,1% | 83,5% (cumulate)",
        "→ soglie 45,9 | 60,6 | 68,5 | 87,4"]),
    "22 → l1, 34 → l1, 42 → l1, 44 → l1, 39 → l1, 37 → l1, 71 → l4, 26 → l1, …",
    ("Si contano: l1 **23**, l2 6, l3 1, l4 1 (su 31)", ["la più votata è **l1** {{✓}}"]),
    ("Reputabile? Solo 2 articoli su 31 in l3–l5 → **no** {{✓}}", []),
], p="X20")

differenza = pagina("Differenza tra ", "1) e 2)", "Non solo media contro maggioranza", [
    ("**Ordine**: 1) media → classe;  2) classe di ogni articolo → voto", []),
    ("**Quote**: 1) dalle testate di training;  2) dagli articoli di training", ["diverse: le testate l4 scrivono meno (36% delle testate, 30% degli articoli)"]),
    ("**Reputabile**: 1) media ≥ 60;  2) maggioranza degli articoli in l3–l5", []),
    ("**Su 226 testate**: classe diversa per 55, giuste 95 contro 96", ["con il voto [[l3 quasi sparisce]] (4 testate contro 28): gli articoli di una testata di mezzo si spargono tra l2 e l4"]),
])

disaccordo = pagina("Differenza tra ", "1) e 2)", "Esempio: americanthinker.com", [
    "49 articoli, voto vero **22,5** (l1)",
    "Voti previsti: 52, 66, 44, 48, 33, 48, 38, 59, …",
    ("**1)** media 50,7 → soglie 50,7 | 55,3 | 60,7 | 86,8 → **l1** {{✓}}", []),
    ("**2)** soglie 46,4 | 57,0 | 64,8 | 83,2 → l1 13, **l2 25**, l3 6, l4 5 → [[l2 ✗]]", []),
    "Gli articoli stanno quasi tutti appena sopra la prima soglia: stessi numeri, classe diversa",
], p="X24")

allena3 = pagina("Regressione per ", "punteggio esatto", "Come si allena?", [
    "Stesso bersaglio: il voto esatto 0-100",
    ("Cambia lo **split**: i fold si fanno per voto, non per classe", [
        "prima le testate l1 avevano voto medio 12 in un fold e 31 in un altro",
        "ora 20-22 in tutti"]),
    ("Si prova a **pesare le testate**: tutti gli articoli, ogni testata uguale, al più 10 o 30 articoli, la media", []),
    "Un classificatore a 5 classi diventa un voto: probabilità × voto medio di ogni classe",
])

valuta3 = pagina("Regressione per ", "punteggio esatto", "Come si valuta?", [
    ("**Nessuna classe** e **nessun riadattamento**: si resta sul numero", []),
    ("Errore = | media prevista − voto vero |", ["lifezette.com: | 41,7 − 0 | = [[42 punti]]"]),
    ("Si fa la media sulle testate: **MAE** (errore medio in punti)", [
        "dire sempre la media del training (68,5): 22,6 punti",
        "**migliore: Mistral 24B (classificatore → voto), 14,7 punti**"]),
    "Le classi di lifezette erano giuste, il numero no: il modello non osa gli estremi",
])

sintesi = pagina("", "In sintesi", "Che cosa succede ai valori del regressore", [
    ("**Allenamento**: sempre sul voto esatto 0-100, nessuna trasformazione", []),
    ("**1) Per classi**: media della testata", ["reputabili: soglia 60, nessun riadattamento", "5 classi: riadattamento con le quote delle testate"]),
    ("**2) Voto degli articoli**: ogni articolo → classe (quote degli articoli) → voto", []),
    ("**3) Punteggio esatto**: media della testata, errore in punti, nessun riadattamento", []),
    "Mai min–max, mai ritagli a 0-100",
])

nuove = [p1, p2, p3, p4, s5, soglie1, soglie2, esempio1, schiaccia, stira, quote, confronto, p6, s7, esempio2, differenza, disaccordo,
         indice(2), allena3, valuta3, sintesi]
corpo = ""
for n, pg in enumerate(nuove, 1):
    pg = re.sub(r'draw:name="[^"]*"', f'draw:name="page{n}"', pg, count=1)
    pg = pg.replace("NOTE", note(n))
    pg = re.sub(r'draw:page-number="\d+"', f'draw:page-number="{n}"', pg)
    corpo += pg
inizio = c.find("<draw:page ")
fine = c.rfind("</draw:page>") + len("</draw:page>")
c = c[:inizio] + corpo + c[fine:]

for _, dest in immagini:
    voce = f' <manifest:file-entry manifest:full-path="{dest}" manifest:media-type="image/png"/>\n'
    if dest not in manifest:
        manifest = manifest.replace("</manifest:manifest>", voce + "</manifest:manifest>")

with zipfile.ZipFile(OUT, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
    for info in z.infolist():
        if info.filename in ("mimetype",):
            continue
        dati = z.read(info.filename)
        if info.filename == "content.xml":
            dati = c.encode()
        elif info.filename == "META-INF/manifest.xml":
            dati = manifest.encode()
        zo.writestr(info, dati, compress_type=zipfile.ZIP_DEFLATED)
    for interno, file in dict((d, s) for s, d in immagini).items():
        zo.write(file, arcname=interno)
print("slide:", len(nuove), "→", OUT)
