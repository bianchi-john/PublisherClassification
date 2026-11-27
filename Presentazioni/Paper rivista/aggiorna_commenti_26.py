"""Ventiseiesima revisione (27 novembre) di PaperRivista.odp (72 slide): le appendici dopo la F (G, H, I, J, K, L-N, O)
nello stile della F, che John ha approvato: elenchi di poche parole, una slide per idea (la critica, come la
controlliamo, il risultato) e le tabelle in una slide tutta loro. Il tema sta solo nel sottotitolo della prima
slide di ogni appendice; le tabelle portano il loro numero nel paper.
Prima: python figure/schermate.py (tab_splits, tab_anon, tab_reading, tab_orientation). Uso:
    python aggiorna_commenti_26.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "yc"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 72, len(p)
primo_g = next(i for i, x in enumerate(p) if "Appendice G" in S.titolo_di(x))
assert all(S.titolo_di(x).split(">")[-1].strip().startswith("Appendice") for x in p[primo_g:]), "dopo la G solo appendici"


def solo_tabella(nome):
    """La schermata di una tabella senza didascalia: dal primo filetto (la riga con il tratto scuro continuo più lungo)
    in giù, tagliata anche ai lati sul contenuto."""
    a = np.array(Image.open(SCH / f"{nome}.png").convert("L")) < 128

    def tratto(riga):                                # il tratto scuro continuo più lungo di una riga
        d = np.diff(np.concatenate([[0], riga.astype(int), [0]]))
        return (np.where(d == -1)[0] - np.where(d == 1)[0]).max() if riga.any() else 0

    tratti = [tratto(r) for r in a]
    alto = next(y for y, t in enumerate(tratti) if t >= 0.9 * max(tratti))
    sotto = a[alto:]
    col, rig = np.where(sotto.any(0))[0], np.where(sotto.any(1))[0]
    uscita = SCH / f"solo_{nome}.png"
    Image.open(SCH / f"{nome}.png").crop((max(col.min() - 6, 0), alto, col.max() + 6, alto + rig.max() + 6)).save(uscita)
    return uscita


def slide(lettera, sotto, voci):
    return S.pagina(f"Appendice {lettera}", sotto, voci, 24, alto_testo=11.6)


def tabella(lettera, sotto, nome):
    """Solo la tabella, la più grande possibile."""
    img = solo_tabella(nome)
    w, h = Image.open(img).size
    largo = min(24.0, 11.6 * w / h)
    return S.pagina(f"Appendice {lettera}", sotto, [],
                    extra=S.immagine(img, (28 - largo) / 2, 3.6 + (11.6 - largo * h / w) / 2, largo=largo))


appendici = [
    slide("G", "«Solo il nome»: la critica", [
        "«I modelli grandi **conoscono già** le testate»",
        "«Le riconoscono, non leggono il testo»",
        "Le testate famose sono su Wikipedia",
    ]),
    slide("G", "Come la controlliamo", [
        "Al modello solo il **nome di dominio**",
        "Es.: example.com, nessun articolo",
        "Stesso vettore, stessa ridge, stessi fold",
    ]),
    slide("G", "Il risultato", [
        "Dal solo nome: ρ **0,35–0,39**",
        "Dal testo: ρ **0,71–0,74**",
        "Il voto viene dagli **articoli**",
        "96% degli articoli: **dopo** il rilascio dei modelli",
    ]),
    slide("H", "Altri split: la critica", [
        "«Lo split è stato **fortunato**?»",
        "226 testate: poche testate spostano il confronto",
    ]),
    slide("H", "Come la controlliamo", [
        "Altri **4 split**, fatti allo stesso modo",
        "Stessi modelli, stesse impostazioni",
    ]),
    slide("H", "Il risultato", [
        "Mistral batte TF-IDF in **ogni** split: 2,0–2,3 punti",
        "Lo split del paper: il vantaggio **più piccolo**",
        "Errore di ogni modello: varia meno di **mezzo punto**",
    ]),
    tabella("H", "Tabella 11", "tab_splits"),
    slide("I", "Testate sorelle: la critica", [
        "«Le sorelle sono davvero un **problema**?»",
        "«Si somigliano davvero?»",
    ]),
    slide("I", "Come la controlliamo", [
        "Quante **copie** condividono",
        "Che **voti** hanno",
        "Quanto aiutano in **training**",
    ]),
    slide("I", "Il risultato", [
        "Testate con copie in comune: **98%** stesso proprietario",
        "**23** testate su 31: stesso voto delle sorelle",
        "Sorelle in training: errore giù di **7–12** punti",
    ]),
    slide("J", "Anonimizzazione a regole: la critica", [
        "«I risultati dipendono dall'**anonimizzazione**?»",
        "GLiNER sbaglia qualche nome",
    ]),
    slide("J", "Come la controlliamo", [
        "Tutta la pipeline rifatta con una versione **precedente**",
        "6 regole, niente GLiNER",
        "Tolgono solo la testata e il suo staff",
    ]),
    slide("J", "Il risultato", [
        "Modelli linguistici: cambiano meno di **0,25** punti",
        "TF-IDF: perde un po' (usa i nomi come parole)",
    ]),
    tabella("J", "Tabella 12", "tab_anon"),
    slide("K", "Boilerplate: la critica", [
        "«I modelli riconoscono la testata dalle **righe fisse**»",
        "Newsletter, donazioni, disclaimer",
    ]),
    slide("K", "Come la controlliamo", [
        "Righe fisse: **3%** delle parole",
        "TF-IDF riaddestrato **senza** di loro",
        "1.000 articoli con e senza",
        "Mistral scomposto **token per token**",
    ]),
    slide("K", "Il risultato", [
        "TF-IDF senza righe fisse: **17,5** contro 17,8",
        "Mistral: le righe fisse lo spostano di **~2** punti",
        "Il voto viene dal **corpo** degli articoli",
    ]),
    tabella("K", "Tabella 13", "tab_reading"),
    slide("L, N", "Le stesse testate: la critica", [
        "«Scelte **adattate** alle 226 testate?»",
        "19 tentativi prima, sulle **stesse** testate",
    ]),
    slide("L, N", "Come la controlliamo", [
        "Elenchiamo **tutto** quello che abbiamo provato",
        "Dentro i fold solo: penalità e layer",
        "Errori per livello, modello per modello",
    ]),
    slide("L, N", "Il risultato", [
        "Stabili sui **5 split**",
        "Anonimizzazione rifatta: meno di mezzo punto",
        "Non è una replica su testate **nuove**: lo diciamo",
    ]),
    slide("O", "Orientamento politico: la critica", [
        "«I modelli trattano diversamente **destra** e **sinistra**?»",
        "NewsGuard accusato di essere più severo con la destra",
    ]),
    slide("O", "Come la controlliamo", [
        "NewsGuard: solo Left, Right o vuoto",
        "Confronto giusto solo **a parità di voto**",
        "Contiamo le testate con lo **stesso voto**",
    ]),
    slide("O", "Il risultato", [
        "Sinistra: voto medio **80**; destra: **54**",
        "Livello 0–39: **3** di sinistra su 35",
        "Voto 100: **4** di destra su 39",
        "**Nessuna conclusione** possibile",
    ]),
    tabella("O", "Tabella 15", "tab_orientation"),
]
nuove = p[:primo_g] + appendici

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 72)")
