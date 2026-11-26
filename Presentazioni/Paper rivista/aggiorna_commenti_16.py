"""Sedicesima revisione del 26 novembre di PaperRivista.odp (60 slide): l'appendice D rifatta come nel paper («How the
dataset compares with NewsGuard»): la critica (un campione che non è la realtà: tolti i casi difficili, quasi tutto USA),
il confronto con NewsGuard (Tabella tab:population) e la risposta (nessuna selezione sul voto, i casi tolti mancano
anche al training, gli errori per livello, lingua e paese).
Prima: compilare il paper e  python figure/schermate.py  (tab_population.png). Uso:
    python aggiorna_commenti_16.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zs"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 60, len(p)
d = [i for i, x in enumerate(p) if "Appendice D" in S.titolo_di(x)]
assert d == [33, 34, 35], d


def D(sotto, voci):
    return S.pagina("Appendice D", sotto, voci, 20, alto_testo=11.6)


appendice_d = [
    D("Il dataset e NewsGuard: la critica", [
        "«Avete costruito un **campione** che non è la realtà delle testate valutate»",
        "In particolare: la pulizia avrebbe tolto i **casi difficili**, le testate meno affidabili, quindi gli errori "
        "sarebbero **ottimisti**",
        "E le testate sono quasi tutte degli **Stati Uniti**",
    ]),
    S.con_grafico("Appendice D", "Il dataset e NewsGuard: i dati", [
        "Ci sono **tutti** i livelli: 35 testate sotto 40, 39 con voto 100, 45 voti diversi",
        "I due livelli più alti pesano come in tutto NewsGuard (35% e 17%, contro 35% e 14%)",
        "Le meno affidabili sono **meno**: 15% del dataset nel livello più basso, contro 33% di tutto NewsGuard e 39% "
        "delle politiche",
    ], SCH / "tab_population.png", 20, largo_max=24.0),
    D("Il dataset e NewsGuard: la risposta", [
        "**Nessuna scelta sul voto**: i filtri guardano il testo (leggibile, inglese, lunghezza, data, copie, almeno 10 "
        "articoli) e la proprietà, mai il voto",
        "Le meno affidabili escono in due passi: **619** senza articoli (siti offline, robot bloccati; voto medio 41) e "
        "**170** solo copie (voto medio 37), quasi tutte la rete da 164 siti",
        "Toglierle dal test toglie casi difficili, ma le toglie anche dal **training**: i modelli imparano la parte bassa "
        "della scala da meno esempi, e questo va **contro** di noi",
        "Gli errori per livello mostrano come va sulle meno affidabili: **31** punti per Mistral nel livello 0–39",
        "Solo inglese: **79%** USA, contro 73% delle politiche e 74% di tutto NewsGuard; i risultati valgono per questa "
        "popolazione, e lo diciamo",
    ]),
]
nuove = p[:33] + appendice_d + p[36:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide")
