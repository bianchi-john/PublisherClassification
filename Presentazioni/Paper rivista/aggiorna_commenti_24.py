"""Ventiquattresima revisione (27 novembre) di PaperRivista.odp (63 slide): la slide «Il dataset e NewsGuard: la
risposta» dell'appendice D divisa in cinque slide a elenchi di poche parole (nessuna scelta sul voto, dove escono le
meno affidabili, lo ammettiamo, due effetti opposti, lingua e paese). Le altre slide di John restano come sono.
Uso:
    python aggiorna_commenti_24.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "ya"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 63, len(p)
i = next(k for k, x in enumerate(p) if "Il dataset e NewsGuard: la risposta" in S.testo(x))


def D(sotto, voci):
    return S.pagina("Appendice D", sotto, voci, 24, alto_testo=11.6)


risposta = [
    D("La risposta: nessuna scelta sul voto", [
        "Filtri sul **testo** e sulla **proprietà**",
        "Mai sul **voto**",
        "Hanno tolto quello che hanno tolto",
    ]),
    D("La risposta: dove escono le meno affidabili", [
        "**619** senza articoli: voto medio 41",
        "**170** solo copie: voto medio 37",
        "Le copie: quasi tutte la rete da **164** siti",
    ]),
    D("La risposta: lo ammettiamo", [
        "Togliamo **molti voti bassi**",
        "Livello più basso: **15%** da noi, **33%** in NewsGuard",
        "Proprio lì i modelli sbagliano di più: **31** punti (Mistral)",
    ]),
    D("La risposta: due effetti opposti", [
        ("Nel **test**: meno casi difficili", ["l'errore medio scende"]),
        ("Nel **training**: meno esempi della parte bassa", ["l'errore lì sale"]),
    ]),
    D("La risposta: lingua e paese", [
        "Solo articoli in **inglese**",
        "Testate: **79%** USA, **10%** Regno Unito, **5%** Canada",
        "Altre lingue e paesi: **non sappiamo**",
    ]),
]
nuove = p[:i] + risposta + p[i + 1:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 63)")
