"""Trentaduesima revisione (27 novembre) di PaperRivista.odp (77 slide): l'appendice K in più slide, che spiegano che cosa
fa già la pulizia (trafilatura, una pagina alla volta), che cosa resta (righe ripetute dentro il corpo degli articoli),
come le troviamo (Code/Pipeline/boilerplate.py), come controlliamo il loro effetto e il risultato. La slide con la tabella
resta quella di prima.
Uso:
    python aggiorna_commenti_32.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "yh"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
k = [i for i, x in enumerate(p) if "Appendice K" in S.titolo_di(x)]
assert len(k) == 4 and re.search(r"Tabella \d+", S.testo(p[k[-1]])), k
tabella = p[k[-1]]


def K(sotto, voci):
    return S.pagina("Appendice K", sotto, voci, 24, alto_testo=11.6)


appendice_k = [
    K("Boilerplate: la critica", [
        "«I modelli riconoscono la testata dalle **righe fisse**»",
        "Newsletter, donazioni, disclaimer",
        "Sarebbe come riconoscerla dal **nome**",
    ]),
    K("Che cosa fa già la pulizia", [
        "Trafilatura: tiene solo il **corpo** dell'articolo",
        "Via menu, barre laterali, piè di pagina",
        "Via commenti, tabelle, immagini, link",
        "Legge **una pagina alla volta**",
    ]),
    K("Che cosa resta", [
        "Righe che la testata mette **dentro** ogni articolo",
        "«Iscriviti alla newsletter», «sostienici», disclaimer, didascalie",
        "Trafilatura non lo sa: non vede gli **altri** articoli",
    ]),
    K("Come le troviamo", [
        "Confrontiamo gli articoli della **stessa testata**",
        ("Boilerplate = una riga (cifre ignorate) che compare", [
            "in almeno **3** articoli",
            "e in almeno il **5%** degli articoli della testata",
        ]),
        "Trovate: **1.040** righe in **171** testate su 226",
        "**3%** delle parole, nel 49% degli articoli",
    ]),
    K("Come la controlliamo", [
        "TF-IDF riaddestrato **senza** quelle righe",
        "1.000 articoli: previsioni **con** e **senza**",
        "Mistral scomposto **token per token**",
    ]),
    K("Il risultato", [
        "TF-IDF senza righe fisse: **17,5** contro 17,8",
        "Mistral: le righe fisse lo spostano di **~2** punti",
        "Il voto viene dal **corpo** degli articoli",
    ]),
    tabella,
    K("Quindi la pulizia regge", [
        "Dopo la pulizia resta solo il **3%** di righe fisse",
        "Toglierle cambia **pochissimo**",
        "Il controllo **completa** la pulizia, non la smentisce",
    ]),
]
nuove = p[:k[0]] + appendice_k + p[k[-1] + 1:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano {len(p)})")
