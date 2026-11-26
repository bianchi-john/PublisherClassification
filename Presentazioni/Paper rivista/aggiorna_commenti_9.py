"""Nona revisione del 26 novembre di PaperRivista.odp (64 slide), sul file di John: nell'appendice B si dice
chiaramente che il bootstrap è uno solo, già nei risultati (gli stessi 10.000 ripescaggi danno gli intervalli delle
tabelle e i confronti con TF-IDF, Code/Pipeline/paper.py), non un calcolo in più.
Uso:
    python aggiorna_commenti_9.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zk"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 64, len(p)
assert "Il bootstrap, lo stesso degli intervalli" in S.testo(p[27])
assert "Il p-value" in S.testo(p[28])

p[27] = S.pagina("Appendice B", "Il bootstrap: uno solo, già nei risultati", [
    "**Non** lo rifacciamo: il bootstrap è **uno solo**, calcolato **una volta**, insieme alla Tabella 4",
    "Ripeschiamo le 226 testate 10.000 volte (con ripetizione) e su ogni ripescaggio calcoliamo le metriche di "
    "**tutti** i modelli",
    ("Da quei 10.000 valori escono due cose:", [
        "gli **intervalli** al 95% tra parentesi nelle tabelle, es. MAE di Mistral 15,8 [14,3; 17,4]",
        "i **confronti** con TF-IDF: sugli stessi ripescaggi, la differenza di MAE di Mistral − TF-IDF, −2,0 "
        "[−2,9; −1,2]; da lì il p e l'asterisco",
    ]),
    "Se Mistral è meglio in **quasi tutti** i ripescaggi, la differenza non dipende dalle testate capitate",
], 20, alto_testo=11.6)
p[28] = S.riscrivi(p[28], "Contiamo i ripescaggi", "Sugli **stessi** 10.000 ripescaggi contiamo quelli in cui la "
                   "differenza va **dall'altra parte**, cioè TF-IDF meglio di Mistral", 20)

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
p = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
               'draw:text-style-name="P2"') for x in p]
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: {len(p)} slide")
