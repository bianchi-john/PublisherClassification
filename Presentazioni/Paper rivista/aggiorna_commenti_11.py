"""Undicesima revisione del 26 novembre di PaperRivista.odp (66 slide): l'appendice B (confronti multipli) torna una slide
sola, come l'appendice B del paper, ridotta a un paragrafo (opzione A scelta da John).
Uso:
    python aggiorna_commenti_11.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zm"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 66, len(p)
b = [i for i, x in enumerate(p) if "Appendice B" in S.titolo_di(x)]
assert b == list(range(26, 39)), b

appendice_b = S.pagina("Appendice B", "Confronti multipli", [
    "La critica: con tanti confronti (24 nella Tabella 4) qualcuno può sembrare significativo **per caso**",
    "Già nel paper: i p sono corretti con **Holm**, che alza la soglia quando i confronti sono tanti; l'asterisco "
    "della Tabella 4 è **dopo** la correzione",
    "Dopo la correzione l'**errore** più basso dei modelli grandi **resta** significativo (Mistral −2,0, "
    "Qwen3-Embedding-8B −1,5), e così l'**ordine** con 1 o 5 articoli per testata",
    "L'**ordine** con tutti gli articoli (Mistral +0,06) **non** lo è, e lo scriviamo",
], 20, alto_testo=11.6)
nuove = p[:26] + [appendice_b] + p[39:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 66)")
