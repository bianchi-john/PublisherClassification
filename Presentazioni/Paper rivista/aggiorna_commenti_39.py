"""Trentanovesima revisione (27 novembre) di PaperRivista.odp: in fondo, dopo le appendici, una slide sul boilerplate con la
frase aggiunta ai limiti del paper, e nient'altro.
Uso:
    python aggiorna_commenti_39.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "yn"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert "Appendice N" in S.titolo_di(p[-1])

boilerplate = S.pagina("Boilerplate", "Come l'abbiamo affrontato: una frase nei limiti del paper", [
    "«The anonymization leaves the style, some brand names, the social-media handles **and the phrases that an outlet "
    "repeats in many of its articles, such as invitations to subscribe**, and the ownership groups are those recorded by "
    "NewsGuard.»",
], 24, alto_testo=11.6)
p.append(boilerplate)

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
p = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
               'draw:text-style-name="P2"') for x in p]
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: {len(p)} slide")
