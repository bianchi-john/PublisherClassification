"""Ventiduesima revisione (27 novembre) di PaperRivista.odp (60 slide): una slide breve per l'appendice C (l'anonimizzazione
in dettaglio), così le lettere delle appendici nel deck sono quelle del paper e non c'è il buco tra B e D. Il contenuto
non si ripete: è nelle slide iniziali sull'anonimizzazione.
Uso:
    python aggiorna_commenti_22.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zy"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 60, len(p)
primo_d = next(i for i, x in enumerate(p) if "Appendice D" in S.titolo_di(x))
assert "Appendice B" in S.titolo_di(p[primo_d - 1])

appendice_c = S.pagina("Appendice C", "L'anonimizzazione in dettaglio", [
    "Nel paper: i passaggi (prima le regole, poi GLiNER), le impostazioni di GLiNER, quante sostituzioni fa ogni "
    "passaggio e la Figura 8",
    "Qui non la ripetiamo: è nelle slide iniziali sull'**anonimizzazione**",
], 20, alto_testo=11.6)
nuove = p[:primo_d] + [appendice_c] + p[primo_d:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 60)")
