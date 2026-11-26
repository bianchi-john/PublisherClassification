"""Diciottesima revisione del 26 novembre di PaperRivista.odp (60 slide), sul file di John:
- appendice D: una slide con la misura della differenza tra il dataset e NewsGuard (distanza di variazione totale tra
  le distribuzioni dei livelli, 17%, calcolata passo per passo) e i due punti che John non capiva riscritti;
- appendici E ed F separate: E (impostazioni) in due slide, F (il prompt) con il contenuto di prima, in attesa di
  decidere con John come presentarla.
Uso:
    python aggiorna_commenti_18.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zu"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 60, len(p)
assert "Il dataset e NewsGuard: la risposta" in S.testo(p[35])
assert [i for i, x in enumerate(p) if "Appendice E, F" in S.titolo_di(x)] == [36, 37, 38]


def pag(titolo, sotto, voci):
    return S.pagina(titolo, sotto, voci, 20, alto_testo=11.6)


distanza = pag("Appendice D", "Il dataset e NewsGuard: quanto è diverso", [
    "Quote per livello (0–39 · 40–59 · 60–74 · 75–99 · 100): nostro dataset 15,5 · 19,0 · 13,3 · 35,0 · 17,3%; tutto "
    "NewsGuard 32,6 · 12,4 · 5,7 · 35,0 · 14,3%",
    "Nel livello più basso ci mancano 32,6 − 15,5 = **17,2** punti percentuali",
    "Negli altri ne abbiamo in più 6,6 + 7,6 + 0 + 3,0 = **17,2**",
    "Quindi basterebbe spostare il **17%** delle nostre testate nel livello più basso per avere la distribuzione di "
    "NewsGuard: è la **distanza di variazione totale** (0% = uguali, 100% = niente in comune)",
    "Per confronto: le testate politiche di NewsGuard, prese tutte, distano **23%** da tutto NewsGuard",
])
p[35] = S.togli_punto(p[35], "coincidono per l'")          # la distanza ha ora la sua slide
p[35] = S.riscrivi(p[35], "Solo inglese:", "Solo **inglese**: testate quasi tutte di paesi anglofoni (79% USA, 10% "
                   "Regno Unito, 5% Canada); non sappiamo se i risultati valgono altrove", 20)

appendice_e = [
    pag("Appendice E", "Impostazioni: la critica", [
        "«Avete scelto le impostazioni dei modelli guardando le testate di **test**?»",
        "Se fosse così, i risultati sarebbero **ottimisti**",
    ]),
    pag("Appendice E", "Impostazioni: la risposta", [
        "**No**: la penalità della ridge e il layer di Mistral sono scelti su **3 fold interni** delle sole testate di "
        "training, in ogni fold",
        "Tutte le altre impostazioni sono **fisse** e scritte nell'appendice E: lo studio si può rifare",
        "Unica eccezione, dichiarata: le impostazioni di ModernBERT vengono dai tentativi precedenti sulle stesse "
        "testate (appendice N); ModernBERT comunque non batte TF-IDF",
    ]),
]


def a_f(pagina):
    pagina = pagina.replace("Appendice E, F", "Appendice F")
    return pagina.replace("Impostazioni e prompt", "Il prompt")


appendice_f = [a_f(p[i]) for i in (36, 37, 38)]
appendice_f[0] = S.togli_punto(appendice_f[0], "Avete scelto le impostazioni")
appendice_f[0] = S.togli_punto(appendice_f[0], "Nel primo caso")
appendice_f[1] = S.togli_punto(appendice_f[1], "Penalità della ridge")
appendice_f[1] = S.togli_punto(appendice_f[1], "ModernBERT: impostazioni fisse")
appendice_f[2] = S.togli_punto(appendice_f[2], "Tutte le impostazioni sono")

nuove = p[:35] + [distanza, p[35]] + appendice_e + appendice_f + p[39:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 60)")
