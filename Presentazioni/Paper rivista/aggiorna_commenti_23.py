"""Ventitreesima revisione (27 novembre) di PaperRivista.odp (61 slide): l'appendice B a elenchi di poche parole, una
slide per idea, con all'inizio a che cosa serve (la Tabella 4 dà solo il risultato, l'appendice B dice come).
Uso:
    python aggiorna_commenti_23.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zz"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 61, len(p)
b = [i for i, x in enumerate(p) if "Appendice B" in S.titolo_di(x)]
assert b == [25, 26, 27], b


def B(sotto, voci):
    return S.pagina("Appendice B", sotto, voci, 24, alto_testo=11.6)


appendice_b = [
    B("A che cosa serve", [
        "La Tabella 4 dà solo il **risultato**: asterisco sì o no",
        "L'appendice B dice **come** lo otteniamo",
        "Serve a chi vuole **controllare** o **rifare** i conti",
    ]),
    S.con_grafico("Appendice B", "La Tabella 4", [
        "**Parentesi**: incertezza dovuta alle testate",
        "**Asterisco**: meglio di TF-IDF, non per caso",
    ], SCH / "solo_tab_main.png", 24, largo_max=26.0),
    B("Come otteniamo le parentesi", [
        "**10.000** ripescaggi delle 226 testate",
        "Ogni volta: le metriche ricalcolate",
        "Parentesi: il **95%** centrale dei valori",
    ]),
    B("Come otteniamo l'asterisco", [
        "Stessi ripescaggi: differenza con **TF-IDF**",
        "**p**: quanto spesso vince TF-IDF",
        "**Holm**: correzione dei p",
        "Asterisco se il p corretto < **0,05**",
    ]),
    B("La critica", [
        "**24** confronti nella Tabella 4",
        "Con tanti confronti, qualcuno esce **per caso**",
    ]),
    B("La risposta", [
        "**Holm**: soglia più severa con tanti confronti",
        "Tre gruppi: Tabella 4, curve, testo",
        ("Esempio: ρ di Mistral, +0,06", [
            "da solo: significativo",
            "con Holm: **no** → niente asterisco",
        ]),
    ]),
    B("Il risultato", [
        "Errore più basso: **significativo**",
        "Ordine con 1 o 5 articoli: **significativo**",
        "Ordine con tutti gli articoli: **no**, e lo scriviamo",
        "Misura principale del paper: l'**errore**",
    ]),
]
nuove = p[:25] + appendice_b + p[28:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 61)")
