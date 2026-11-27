"""Trentaquattresima revisione (27 novembre) di PaperRivista.odp (73 slide): le appendici K (errori per livello),
M (che cos'altro abbiamo provato) e N (orientamento politico) separate e spiegate in più slide, a elenchi brevi.
Numeri: Dataset/tentativi/20_FromScratch/paper_errors.csv (errori per livello) e il testo del paper.
Prima: figure/schermate/solo_errors.png (Figura 7 senza didascalia). Uso:
    python aggiorna_commenti_34.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "yi"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
AUX = QUI.parents[1] / "Paper" / "Journal" / "main.aux"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
numero = dict(re.findall(r"\\newlabel\{([a-z]+:[^}]+)\}\{\{([^}]+)\}", AUX.read_text()))

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
km = [i for i, x in enumerate(p) if "Appendice K, M" in S.titolo_di(x)]
n = [i for i, x in enumerate(p) if "Appendice N" in S.titolo_di(x)]
assert len(km) == 3 and len(n) == 4 and n[0] == km[-1] + 1 and n[-1] == len(p) - 1, (km, n)
tabella_n = p[n[-1]]
assert f"Tabella {numero['tab:orientation']}" in S.testo(tabella_n)


def slide(lettera, sotto, voci):
    return S.pagina(f"Appendice {lettera}", sotto, voci, 24, alto_testo=11.6)


figura = SCH / "solo_errors.png"
w, h = Image.open(figura).size
largo = min(24.0, 11.6 * w / h)
appendice_k = [
    slide("K", "Errori per livello: la domanda", [
        "L'errore medio (Mistral **15,8**) è una media su tutti i livelli",
        "**Dove** sbagliano i modelli?",
    ]),
    S.pagina("Appendice K", f"Figura {numero['fig:errors']}", [],
             extra=S.immagine(figura, (28 - largo) / 2, 3.6 + (11.6 - largo * h / w) / 2, largo=largo)),
    slide("K", "Errore con segno (+ = voto troppo alto)", [
        "0–39: Mistral **+31**, TF-IDF **+36**",
        "40–59: +9 e +11",
        "60–74: −3 e −2",
        "75–99: −11 e −13",
        "100: **−19** e **−25**",
    ]),
    slide("K", "Errore di Mistral per livello", [
        "0–39: **31,0** punti",
        "40–59: 11,5 · 60–74: **8,5** · 75–99: 12,7",
        "100: **18,9** punti",
        "TF-IDF peggio agli estremi: 36,5 e 25,4",
    ]),
    slide("K", "La compressione", [
        "Le previsioni stanno **strette al centro**",
        "1 punto di voto vero → **0,19–0,37** punti di previsione",
        "Zero-shot: solo **0,09**",
        "Uguale per articolo e per testata: la media **non** la corregge",
    ]),
    slide("K", "Che cosa vuol dire", [
        "Testate pessime **sopravvalutate**, perfette **sottovalutate**",
        "L'**ordine** delle testate resta buono",
        f"Una retta stimata sul training allarga la scala: errore −1/−3 punti (Tabella {numero['tab:robust']})",
    ]),
]
appendice_m = [
    slide("M", "Che cos'altro abbiamo provato: la critica", [
        "«Avete **adattato** le scelte alle 226 testate?»",
        "19 tentativi prima, sulle **stesse** testate",
        "E ne abbiamo visto i risultati",
    ]),
    slide("M", "Che cosa abbiamo provato (1/2)", [
        "**Classificazione** nei 5 livelli invece del voto",
        "**Gemma-27B** congelato: scartato (numeri fuori scala sulla nostra GPU)",
        "Mistral **strato 40**: ora scelto nei fold interni",
        "**DeBERTa** addestrato; impostazioni di ModernBERT",
    ]),
    slide("M", "Che cosa abbiamo provato (2/2)", [
        "**14 regressori** sui vettori di Qwen e Mistral",
        "Es.: rete neurale, attenzione sugli articoli, pesi, mixup",
        "Il migliore: solo **−1** punto rispetto alla ridge",
        "Voti rimappati sui quantili; anonimizzazione a regole (J)",
    ]),
    slide("M", "La risposta", [
        "Lo diciamo **tutto** in appendice M",
        "Scelti dentro i fold: solo **penalità** e **layer**",
        "Il vantaggio su TF-IDF **non dipende** dal regressore",
        "Stabili sui **5 split** (H), ma non è un test su testate **nuove**",
    ]),
]
appendice_n = [
    slide("N", "Orientamento politico: la critica", [
        "«I modelli trattano diversamente **destra** e **sinistra**?»",
        "NewsGuard accusato di essere più severo con la destra",
        "Un'analisi dei suoi dati **non** trova questo bias",
    ]),
    slide("N", "Come la controlliamo", [
        "NewsGuard: solo Left, Right o campo vuoto",
        "Contiamo le testate per **livello** e **orientamento**",
        "Guardiamo quante hanno lo **stesso voto**",
    ]),
    slide("N", "I dati", [
        "Voto medio: sinistra **80**, destra **54**, nessuno **80**",
        "Livello 0–39: **3** di sinistra su 35",
        "Voto 100: **4** di destra su 39",
        "Solo **14** dei 44 voti hanno testate di entrambe le parti",
    ]),
    tabella_n,
    slide("N", "Perché non possiamo confrontare", [
        "Destra e sinistra hanno voti **diversi**",
        "Tutti i modelli sovrastimano i voti bassi, sottostimano gli alti",
        "Errori diversi tra le parti possono venire **solo** da questo",
        "A parità di voto: troppe **poche** testate",
    ]),
    slide("N", "Il risultato", [
        "Solo l'orientamento: ρ **0,36**, errore 18,5 (costante 22,1)",
        "I modelli di testo ordinano **molto meglio**",
        "Su destra e sinistra: **nessuna conclusione**",
        "Servirebbero testate bilanciate a ogni livello",
    ]),
]
nuove = p[:km[0]] + appendice_k + appendice_m + appendice_n

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano {len(p)})")
