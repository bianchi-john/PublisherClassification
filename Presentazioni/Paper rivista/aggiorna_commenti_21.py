"""Ventunesima revisione (27 novembre) di PaperRivista.odp (64 slide): l'appendice B semplificata, da 7 slide a 3
(che cosa c'è nel paper, la critica e la risposta, il risultato), con concetti più veloci.
Uso:
    python aggiorna_commenti_21.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zx"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 64, len(p)
b = [i for i, x in enumerate(p) if "Appendice B" in S.titolo_di(x)]
assert b == list(range(25, 32)), b

appendice_b = [
    S.con_grafico("Appendice B", "Confronti multipli: che cosa c'è nel paper", [
        "Tra **parentesi**: dove potrebbe stare il valore con altre testate (bootstrap: 10.000 ripescaggi delle testate)",
        "**Asterisco**: migliore di TF-IDF in modo significativo, cioè non per caso, **dopo** la correzione di Holm",
    ], SCH / "solo_tab_main.png", 20, largo_max=26.0),
    S.pagina("Appendice B", "Confronti multipli: la critica e la risposta", [
        "**La critica**: confrontiamo con TF-IDF 6 modelli su 4 colonne, cioè 24 confronti; con tanti confronti, "
        "qualcuno sembra significativo **per caso**",
        "**La risposta**, già nel paper: la correzione di **Holm** alza la soglia quando i confronti sono tanti",
        "Esempio: il ρ di Mistral (+0,06 su TF-IDF) da solo sembrerebbe significativo; con la correzione non lo è più, "
        "e per questo non ha l'asterisco",
    ], 20, alto_testo=11.6),
    S.pagina("Appendice B", "Confronti multipli: il risultato", [
        "**Significativo**: l'errore (MAE) più basso dei modelli grandi (Mistral −2,0, Qwen3-8B −1,5), e l'ordine delle "
        "testate con 1 o 5 articoli",
        "**Non significativo**: l'ordine con tutti gli articoli (+0,06), e lo scriviamo",
        "Non è un problema per il paper: la misura principale è l'errore, e con tutti gli articoli Mistral ordina "
        "comunque meglio di TF-IDF in tutti e 5 gli split",
    ], 20, alto_testo=11.6),
]
nuove = p[:25] + appendice_b + p[32:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 64)")
