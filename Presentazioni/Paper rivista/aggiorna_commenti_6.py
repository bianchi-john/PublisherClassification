"""Sesta revisione del 26 novembre di PaperRivista.odp (55 slide), sul file di John: l'appendice A spiega meglio le tre
metriche in più (RMSE, R², AUC), una slide per metrica con un esempio, al posto della slide unica.
Uso:
    python aggiorna_commenti_6.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zh"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 55, len(p)
assert "Appendice A" in S.titolo_di(p[23]) and "Metriche in più: come la affrontiamo" in S.testo(p[23])


def pagina(sotto, voci):
    return S.pagina("Appendice A", sotto, voci, 20, alto_testo=11.6)


metriche = [
    pagina("Metriche in più: le tre metriche", [
        "Aggiungiamo tre metriche usate da altri lavori, calcolate sulle **stesse previsioni** della Tabella 4",
        "Tutte **per testata**: il voto previsto di una testata è la media dei suoi articoli, perché il compito è il "
        "voto della testata",
        "**RMSE** e **R²** guardano quanto il voto previsto è **vicino** al vero",
        "L'**AUC** guarda se il modello **separa** le testate affidabili dalle altre",
    ]),
    pagina("Metriche in più: RMSE", [
        "Come il MAE, ma gli errori **grandi pesano di più**: si elevano al quadrato, si fa la media, poi la radice",
        "Un errore di 40 pesa come **16** errori di 10",
        ("Esempio, 4 testate:", [
            "il modello A sbaglia di 10, 10, 10, 10 → MAE **10**, RMSE **10**",
            "il modello B sbaglia di 0, 0, 0, 40 → MAE **10**, RMSE **20**",
            "stesso MAE, ma B fa un errore grosso: l'RMSE lo vede",
        ]),
        "Nei dati: Mistral MAE 15,8 e RMSE **19,6**; TF-IDF 17,8 e **22,0**. L'RMSE è più alto del MAE perché alcune "
        "testate, le meno affidabili, hanno errori grandi",
    ]),
    pagina("Metriche in più: R²", [
        "Quanta parte delle **differenze tra i voti** il modello spiega, rispetto a chi risponde **sempre la media**",
        "**1** = perfetto; **0** = come dare sempre la media; **sotto 0** = peggio della media",
        ("Esempio, 3 testate con voto 20, 50, 80 (media 50):", [
            "sempre 50 → errori 30, 0, 30 → R² **0**",
            "il modello prevede 30, 50, 70 → errori 10, 0, 10; al quadrato 200 contro 1.800 → R² = 1 − 200/1.800 = "
            "**0,89**",
        ]),
        "Nei dati: Mistral **0,48**, TF-IDF **0,34**; la mediana costante **−0,06**, cioè un po' peggio della media",
    ]),
    pagina("Metriche in più: AUC", [
        "Affidabile = voto ≥ **60** (65% delle testate); non serve scegliere una soglia sul voto previsto",
        "Si prendono **tutte le coppie** (una testata affidabile, una no) e si conta quante volte l'affidabile ha il "
        "voto previsto **più alto**",
        ("Esempio: affidabili A (previsto 80) e B (55); non affidabili C (60) e D (40)", [
            "A sopra C ✓, A sopra D ✓, B sopra C ✗, B sopra D ✓ → 3 coppie su 4 → AUC **0,75**",
        ]),
        "**0,5** = a caso (come la costante); **1** = sempre giusto",
        "Nei dati: Mistral **0,88**, TF-IDF **0,86**: presa una testata affidabile e una no, Mistral mette più in alto "
        "quella affidabile 88 volte su 100",
    ]),
]
nuove = p[:23] + metriche + p[24:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 55)")
