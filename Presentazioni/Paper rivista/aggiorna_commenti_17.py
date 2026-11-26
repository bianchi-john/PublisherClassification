"""Diciassettesima revisione del 26 novembre di PaperRivista.odp (60 slide), appendice D come nel paper:
- la slide dei dati mostra solo la tabella (ora con i paesi), senza testo;
- la risposta: nessuna scelta sul voto (filtri sul testo e sulla proprietà, hanno tolto quello che hanno tolto);
  ammettiamo che togliamo molti voti bassi, proprio dove i modelli sbagliano di più; due effetti opposti (meno casi
  difficili nel test, meno esempi nel training); somiglianza con NewsGuard 83%; solo inglese.
Prima: compilare il paper e  python figure/schermate.py  (tab_population.png). Uso:
    python aggiorna_commenti_17.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zt"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 60, len(p)
assert "Il dataset e NewsGuard: i dati" in S.testo(p[34]) and "Il dataset e NewsGuard: la risposta" in S.testo(p[35])


def solo_tabella(nome):
    """La schermata di una tabella senza didascalia: dal primo filetto (una riga quasi tutta scura) in giù."""
    im = Image.open(SCH / f"{nome}.png").convert("L")
    w, h = im.size
    px = im.load()
    riga = next(y for y in range(h) if sum(px[x, y] < 128 for x in range(w)) > 0.85 * w)
    uscita = SCH / f"solo_{nome}.png"
    im.crop((0, riga, w, h)).save(uscita)
    return uscita


tabella = solo_tabella("tab_population")
w, h = Image.open(tabella).size
largo = 26.0
p[34] = S.pagina("Appendice D", "Il dataset e NewsGuard: i dati", [],
                 extra=S.immagine(tabella, (28 - largo) / 2, 3.6 + (11.7 - largo * h / w) / 2, largo=largo))
p[35] = S.pagina("Appendice D", "Il dataset e NewsGuard: la risposta", [
    "**Nessuna scelta sul voto**: abbiamo messo filtri sul testo e sulla proprietà, e hanno tolto quello che hanno tolto",
    "Le meno affidabili escono soprattutto in due passi: **619** senza articoli (voto medio 41) e **170** solo copie "
    "(voto medio 37), quasi tutte la rete da 164 siti",
    "Lo ammettiamo: togliamo **molti voti bassi** (15% nel livello più basso, contro 33% di NewsGuard), ed è proprio lì "
    "che i modelli sbagliano di più (**31** punti per Mistral)",
    "Due effetti opposti: nel test meno casi difficili (l'errore medio scende), nel training meno esempi della parte "
    "bassa (l'errore lì sale). Con questi dati non li possiamo separare",
    "Comunque i livelli del dataset coincidono per l'**83%** con quelli di tutto NewsGuard (le politiche nel complesso: "
    "77%)",
    "Solo inglese: 79% USA, 10% Regno Unito, 5% Canada; i risultati valgono per queste testate",
], 20, alto_testo=11.6)

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
p = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
               'draw:text-style-name="P2"') for x in p]
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: {len(p)} slide")
