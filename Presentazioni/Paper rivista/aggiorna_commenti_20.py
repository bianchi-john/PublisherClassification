"""Ventesima revisione del 26 novembre di PaperRivista.odp (64 slide), appendice F:
- «che cosa gli chiediamo»: tre righe semplici (contesto NewsGuard, perché spieghiamo i segnaposto, un solo numero) e
  sotto l'immagine del prompt vero (figure/schermata_prompt.py → figure/schermate/prompt.png);
- la risposta alla critica più semplice e onesta: sì, il prompt si può migliorare (ne abbiamo provato uno); ma lo
  zero-shot è un riferimento, tutti i modelli addestrati fanno meglio, e lo stesso tipo di modello con una piccola parte
  addestrata sui voti diventa il migliore.
Uso:
    python aggiorna_commenti_20.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zw"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 64, len(p)
assert "Il prompt: com'è fatto e che cosa genera" in S.testo(p[40]) and "Il prompt: la risposta" in S.testo(p[42])


def F(sotto, voci, extra=""):
    return S.pagina("Appendice F", sotto, voci, 20, alto_testo=11.6, extra=extra)


voci = [
    "Prima gli **spieghiamo** che cos'è NewsGuard e i suoi nove criteri, come contesto: non gli chiediamo un voto per "
    "criterio",
    "Poi gli diciamo che nell'articolo i nomi sono **nascosti** ([OUTLET], [AUTHOR], …), così non li prende per parole "
    "strane o errori",
    "Infine chiediamo **un solo numero**: il voto NewsGuard (0–100) del sito che ha pubblicato l'articolo",
]
prompt = SCH / "prompt.png"
w, h = Image.open(prompt).size
largo = min(26.0, 11.8 * w / h)                    # il prompt da solo, il più grande possibile
testo_prompt = S.pagina("Appendice F", "Il prompt: il testo", [],
                        extra=S.immagine(prompt, (28 - largo) / 2, 3.5 + (11.8 - largo * h / w) / 2, largo=largo))
spiegazione = F("Il prompt: che cosa gli chiediamo", voci)
p[42] = F("Il prompt: la risposta", [
    "**Sì**, il prompt si potrebbe migliorare: ne abbiamo provato **uno solo**, e un altro potrebbe andare meglio",
    "Ma lo zero-shot non è uno dei nostri modelli: è il riferimento per «chiedere direttamente» a un modello "
    "linguistico, e le conclusioni del paper non dipendono da lui",
    "Tutti i modelli **addestrati** sui voti, da TF-IDF ai transformer, fanno meglio: errore 15,8–18,8 contro 24,4",
    "E lo **stesso tipo** di modello (Mistral), se invece di chiedergli il voto gli mettiamo sopra una piccola parte "
    "addestrata sui voti, diventa il **migliore** di tutti",
    "Quindi il modello «sa» qualcosa delle testate; senza addestramento non sa **che numero** dare: risponde quasi "
    "sempre 65 o 45",
])

p = p[:40] + [testo_prompt, spiegazione] + p[41:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
p = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
               'draw:text-style-name="P2"') for x in p]
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: {len(p)} slide (erano 64)")
