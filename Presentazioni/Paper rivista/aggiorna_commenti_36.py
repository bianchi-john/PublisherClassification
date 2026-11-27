"""Trentaseiesima revisione (27 novembre) di PaperRivista.odp: nell'appendice M (orientamento) la slide «Il risultato» spiegata
meglio, in due slide: che cos'è la prova «solo l'orientamento» (una baseline della pipeline, paper_main_table.csv) e che
cosa dice il confronto con i modelli che leggono il testo (ρ 0,61-0,74, Tabella 4).
Uso:
    python aggiorna_commenti_36.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "yk"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
i = next(i for i, x in enumerate(p) if "Appendice M" in S.titolo_di(x) and "Solo l'orientamento" in S.testo(x))


def M(sotto, voci):
    return S.pagina("Appendice M", sotto, voci, 24, alto_testo=11.6)


nuove_m = [
    M("Prova: solo l'orientamento, senza testo", [
        "A ogni testata il voto medio delle testate di training con lo **stesso orientamento**",
        "Destra: circa **54**; sinistra e nessuno: circa **80**",
        "Ordina un po' le testate: ρ **0,36**",
        "Sbaglia di **18,5** punti; lo stesso voto a tutte: **22,1**",
        "Solo perché destra e sinistra hanno voti medi **diversi**",
    ]),
    M("Il risultato", [
        "I modelli che leggono il testo ordinano **molto meglio**: ρ **0,61–0,74**",
        "Non si limitano a riconoscere la parte politica",
        "Su destra e sinistra: **nessuna conclusione**",
        "Servirebbero testate bilanciate a ogni livello",
    ]),
]
p = p[:i] + nuove_m + p[i + 1:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
p = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
               'draw:text-style-name="P2"') for x in p]
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: {len(p)} slide; slide {i + 1} divisa in due")
