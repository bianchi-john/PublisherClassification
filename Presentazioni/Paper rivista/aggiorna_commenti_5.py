"""Quinta revisione del 26 novembre di PaperRivista.odp (56 slide), sul file di John: niente più orientamento politico.
Il paper non ne parla più (solo l'appendice O spiega perché con questi dati non si può analizzare), quindi:
- tolte le quattro slide «Orientamento politico»;
- Figura 3 rifatta (il pannello (a) è il numero di testate per livello), Tabella 4, Tabella 8 e Tabella 9 senza
  orientamento;
- nelle appendici niente più orientamento, e tre slide nuove per l'appendice O.
Prima: compilare il paper, python figure/schermate.py e solo_dataset.png senza didascalia. Uso:
    python aggiorna_commenti_5.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zg"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 56, len(p)


def titolo(i, atteso):
    assert atteso in S.titolo_di(p[i]), (i, S.titolo_di(p[i]))
    return p[i]


def riscrivi(i, vecchio, nuovo):
    """Come S.riscrivi, con la dimensione del testo del paragrafo che c'era (20 o 24 punti)."""
    par = next(m.group(0) for m in re.finditer(r"<text:p[ >].*?</text:p>", p[i], flags=re.S) if vecchio in S.testo(m.group(0)))
    stile = re.search(r'text:style-name="([^"]+)"', par).group(1)
    corpo = int(float(re.search(rf'<style:style style:name="{stile}".*?fo:font-size="([\d.]+)pt"', c).group(1)))
    p[i] = S.riscrivi(p[i], vecchio, nuovo, corpo)


def figura_poi_testo(tit, sotto, img, voci, corpo=20):
    """La figura in alto, il testo in fondo (al posto della didascalia)."""
    alto_testo = S.altezza_testo(voci, corpo) + 0.3
    y_testo = 15.4 - alto_testo
    w, h = Image.open(img).size
    largo = min(26.0, (y_testo - 0.2 - 3.4) * w / h)
    extra = S.immagine(img, (28 - largo) / 2, 3.4, largo=largo)
    pag = S.pagina(tit, sotto, voci, corpo, alto_testo=alto_testo, extra=extra)
    return pag.replace('svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline"',
                       f'svg:x="1.471cm" svg:y="{y_testo:.2f}cm" presentation:class="outline"')


# Figura 3: il pannello (a) è il numero di testate per livello
titolo(7, "Il dataset finale")
p[7] = figura_poi_testo("Il dataset finale", "Figura 3", SCH / "solo_dataset.png", [
    "(a) Testate per livello NewsGuard; (b) articoli per testata in ordine crescente: minimo **10**, mediana 112, "
    "massimo 201; (c) voto delle testate in ogni fold",
])
# tabelle senza la baseline e le colonne dell'orientamento
p[11] = S.cambia_immagine(titolo(11, "Risultati: testate"), SCH / "tab_main.png")
p[28] = S.cambia_immagine(titolo(28, "Appendice A"), SCH / "tab_more.png")
p[34] = S.cambia_immagine(titolo(34, "Appendice D"), SCH / "tab_selection.png")
# via l'orientamento dal testo
titolo(25, "Le appendici"); riscrivi(25, "Appendici A–N", "Appendici **A–O**")
titolo(33, "Appendice D"); riscrivi(33, "livelli, orientamento, quota USA", "Per ogni passo: quante escono, voto medio, "
                                     "livelli, quota USA")
titolo(40, "Appendice G"); riscrivi(40, "Il solo nome vale quanto", "Il grosso viene dalla **lettura degli articoli**, "
                                    "non dal nome")
titolo(54, "Appendice L, N"); riscrivi(54, "Errori per livello e orientamento", "Errori per livello, modello per modello")
for i in range(19, 23):
    titolo(i, "Orientamento politico")

# l'appendice O: critica, come, risultato
nuove = p[:19] + p[23:] + [
    S.pagina("Appendice O", "Orientamento politico: la critica", [
        "«I modelli trattano diversamente le testate di **destra** e di **sinistra**?»",
        "NewsGuard stesso è stato accusato di essere più severo con le testate di destra",
    ], 20, alto_testo=11.6),
    S.pagina("Appendice O", "Orientamento politico: come la affrontiamo", [
        "NewsGuard dà solo **Left**, **Right** o campo vuoto",
        "Confrontare gli errori delle due parti ha senso solo **a parità di voto**: tutti i modelli sovrastimano i voti "
        "bassi e sottostimano gli alti",
        "Contiamo quante testate di sinistra e di destra hanno **lo stesso voto** o lo stesso livello",
    ], 20, alto_testo=11.6),
    S.con_grafico("Appendice O", "Orientamento politico: il risultato", [
        "Sinistra: voto medio **80**; destra: **54**. Livello 0–39: **3** testate di sinistra su 35; voto 100: **4** di "
        "destra su 39",
        "Solo **14** dei 44 voti hanno testate di entrambe le parti → **nessuna conclusione**: servirebbero testate "
        "bilanciate per orientamento a ogni livello",
    ], SCH / "tab_orientation.png", 20),
]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 56), {len(S.immagini)} immagini nuove")
