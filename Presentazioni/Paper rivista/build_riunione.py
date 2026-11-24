"""Costruisce PaperRivista.odp, le slide per la riunione sul paper per la rivista, nello stile di FinalState.odp (titolo
rosso, sottotitolo verde, punti brevi, nessuna animazione: John spiega a voce). Le tabelle e le figure sono le
«schermate» del paper, con numero e didascalia: prima  python figure/schermate.py  (dopo aver compilato il paper),
poi  python build_riunione.py"""
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent / "Stato finale"))
import stile_slide as S  # noqa: E402

MODELLO = QUI.parent / "Stato finale" / "FinalState.odp"
OUT = QUI / "PaperRivista.odp"
SCH = QUI / "figure" / "schermate"
S.PREFISSO[0] = "pr"
S.animazione = lambda ids: ""                       # nessuna animazione: tutto il testo subito


def sch(nome):
    return SCH / f"{nome}.png"


def testo(titolo, sottotitolo, voci, corpo=24):
    return S.pagina(titolo, sottotitolo, voci, corpo, alto_testo=11.6)


def grafico(titolo, sottotitolo, nome, voci=(), corpo=20):
    return S.con_grafico(titolo, sottotitolo, list(voci), sch(nome), corpo)


def affiancate(titolo, sottotitolo, nomi, voci=(), corpo=20):
    """Due schermate una accanto all'altra, la più grande possibile."""
    y = 3.6 + S.altezza_testo(list(voci), corpo) + 0.3 if voci else 3.5
    largo, spazio = 13.0, 0.6
    extra, x = "", (28 - 2 * largo - spazio) / 2
    for n in nomi:
        w, h = Image.open(sch(n)).size
        l = min(largo, (15.3 - y) * w / h)
        extra += S.immagine(sch(n), x + (largo - l) / 2, y, largo=l)
        x += largo + spazio
    return S.pagina(titolo, sottotitolo, list(voci), corpo, alto_testo=max(y - 3.4, 0.8), extra=extra)


def appendice(lettera, titolo, critica, come, risultato, nome=None, didascalia=""):
    """La critica, come è fatto (passo per passo), il risultato; poi la schermata da sola nella slide dopo."""
    voci = [f"[[Critica:]] {critica}", ("**Come:**", come), f"**Risultato:** {risultato}"]
    prima = testo(f"Appendice {lettera}", titolo, voci, corpo=20)
    return [prima] if nome is None else [prima, grafico(f"Appendice {lettera}", didascalia or titolo, nome)]


pagine = [
    # ------------------------------------------------------------------ apertura
    S.pagina("Il paper per la rivista", "Stato del lavoro e scelte da fare", [
        "**Traces of Reliability** ((titolo provvisorio))",
        "Bozza completa: 19 pagine + appendici nello stesso PDF",
    ]),
    testo("Come vendiamo il paper", "Una doppia valenza", [
        ("[[Un protocollo di valutazione]]", [
            "testate mai viste in training",
            "gruppi di proprietà separati",
            "deduplicazione tra testate",
            "anonimizzazione (GLiNER)",
            "quanto gonfiano i protocolli più deboli: 6–12 punti",
        ]),
        ("[[Modelli che funzionano]]", [
            "Mistral-24B congelato + ridge: errore **15,8** contro **17,8** di TF-IDF",
            "meglio di TF-IDF in **ogni** condizione, soprattutto con pochi articoli",
        ]),
    ]),
    testo("Il titolo", "Opzioni da scegliere (più corto, anche i modelli)", [
        "Predicting the Reliability of Unseen News Outlets from Anonymized Text",
        "Traces of Reliability: Rating Unseen News Outlets from Their Anonymized Articles",
        "Reading Reliability: Language Models Rate Unseen News Outlets from Anonymized Articles",
        "Can Text Alone Rate a News Outlet? Leakage-Free Reliability Prediction",
        "Leakage-Free News Source Reliability Prediction with Frozen Language Models",
        "((Attuale: Traces of Reliability: What the Anonymized Articles of Unseen News Publishers Reveal about Their "
        "Rating, and How Leakage Inflates It))",
    ], corpo=20),
    testo("Cosa lasciamo (1/2)", "Il protocollo", [
        "Un **protocollo**: testate mai viste, gruppi di proprietà, deduplicazione, anonimizzazione",
        "Un **dataset**: 25.524 articoli, 226 testate politiche",
        "La **misura del leakage**: senza protezioni l'errore scende di **6–12 punti**",
    ]),
    testo("Cosa lasciamo (2/2)", "I modelli e il resto", [
        "**Modelli**: LLM congelati + ridge battono TF-IDF ovunque, soprattutto con pochi articoli",
        "**Errori**: voti compressi al centro, divario di ~9 punti per orientamento",
        "**Risorse aperte**: codice, modelli, previsioni, tool",
    ]),
    grafico("Lavori più vicini", "Tabella 1, nell'introduzione (Sezione 2)", "tab_related"),
    # ------------------------------------------------------------------ dataset
    affiancate("Il dataset", "Tabella 2 e Figura 1, nel paper (Sezione 4)", ["tab_cleaning", "fig_groups"]),
    grafico("Gruppi di proprietà", "Figura 2 (nuova): una mappa per passo di pulizia", "fig_network", [
        "Ogni testata è un punto nello **spazio dei testi** (t-SNE del TF-IDF); testate simili vicine",
        "La rete di **164 siti** (quadrati blu): un grappolo fitto che la deduplicazione **cancella**",
        "Copie dello stesso proprietario 19,4%, di altri 2,7% → dopo la pulizia **0**; all'ultimo passo: nessuna sorella",
    ]),
    testo("Anonimizzazione", "La deleghiamo a uno strumento fatto apposta", [
        "Testate e giornalisti → **[OUTLET]**, **[AUTHOR]**",
        "**GLiNER**: un NER zero-shot, gli diamo le etichette «news outlet» e «journalist»",
        "Niente centinaia di regole scritte a mano",
        "Come **trafilatura** per il testo: estrae l'articolo dall'HTML di qualunque sito, senza regole per sito",
        "Più due regole semplici: link ed e-mail, il nome della testata dal dominio",
    ]),
    grafico("Anonimizzazione", "Che cosa sostituisce (Appendice C)", "fig_anonymization"),
    grafico("Il dataset finale", "Figura 3", "fig_dataset"),
    grafico("La pipeline", "Figura 4", "fig_pipeline"),
    # ------------------------------------------------------------------ esperimenti
    testo("Setup sperimentale", "Sezione 5.1", [
        ("**Cross-validation a 5 fold sulle testate**", [
            "stratificata sul voto",
            "ogni testata valutata una volta, da un modello che non l'ha mai vista",
            "3 fold interni per scegliere penalità ridge e layer di Mistral",
        ]),
        ("**Modelli**", [
            "TF-IDF + ridge",
            "LLM congelati + ridge: Qwen3-Embedding 8B e 0,6B, Mistral-Small-24B, ModernBERT",
            "ModernBERT addestrato; Mistral-Instruct zero-shot",
        ]),
        "**Voto della testata** = media delle previsioni sui suoi articoli",
    ], corpo=20),
    grafico("Risultati: articoli", "Tabella 3, Stage 1", "tab_article"),
    grafico("Risultati: testate", "Tabella 4, Stage 2", "tab_main"),
    testo("Le baseline senza testo", "Le righe sopra la linea nella Tabella 4", [
        ("**no text (training mean)**: a ogni testata lo stesso voto", [
            "la media delle testate di training (circa 68) → errore 22,6",
        ]),
        ("**no text (training median)**: la mediana (75)", [
            "la miglior costante per il MAE → errore 22,1",
        ]),
        ("**orientation only**: la media del training per orientamento", [
            "es. destra → 54, sinistra o nessuna → 80; errore 18,5, ρ 0,36",
        ]),
    ], corpo=20),
    grafico("Voto previsto e voto vero", "Figura 5", "fig_scatter"),
    grafico("Contro TF-IDF", "Tabella 5: la differenza diretta con TF-IDF", "tab_beats"),
    # ------------------------------------------------------------------ tabella 6
    testo("Testate non valutate (1/3)", "Perché la Tabella 6", [
        "Il dataset **non è un campione a caso**",
        "Livello più basso: **15%** nel dataset, **43%** tra le testate non raccolte",
        "Proprio lì i modelli sbagliano di più (~31 punti)",
        "→ l'errore sulle 226 testate è **ottimista**",
    ]),
    testo("Testate non valutate (2/3)", "Come lo stimiamo", [
        "Errore medio **per livello** NewsGuard",
        "**Ripesato** con i livelli delle testate non raccolte (o di tutte le politiche)",
        "Nella stessa tabella: ricalibrazione nei fold di training, senza le 16 testate col marchio nel testo",
    ]),
    grafico("Testate non valutate (3/3)", "Tabella 6", "tab_robust", [
        "Ripesato: Mistral **~20**, TF-IDF **22,9**, costante **34,9**; Mistral resta avanti di 2–2,7",
    ]),
    # ------------------------------------------------------------------ tabella 7
    testo("Quanto conta il protocollo (1/4)", "La domanda (Sezione 5.4)", [
        ("Tre modi per **riconoscere** una testata invece di leggerla:",
         ["la stessa testata in training", "testate dello stesso proprietario in training", "i nomi nel testo"]),
        "Quanto vale ciascuna protezione?",
    ]),
    testo("Quanto conta il protocollo (2/4)", "I protocolli più deboli", [
        "**Article split**: fold sugli articoli, la testata di test è anche in training",
        "**Sisters**: +89 testate sorelle in training → 31 testate di test hanno una sorella",
        "**Original text**: nessuna anonimizzazione",
        "**Rules only**: solo le regole, senza GLiNER",
        "Sempre **gli stessi articoli e fold di test**; ognuno confrontato con chi ha una protezione in più",
    ], corpo=20),
    grafico("Quanto conta il protocollo (3/4)", "Tabella 7", "tab_ladder"),
    testo("Quanto conta il protocollo (4/4)", "Che cosa dice", [
        "Split per articolo: errore **−5,6 / −7,4** punti",
        "Sorelle in training: **−7 / −12** sulle 31 testate con una sorella",
        "Nomi: al massimo **0,4** punti",
        "→ senza separare testate e proprietari si misura il **riconoscimento**, 3–6 volte il distacco tra modelli",
    ], corpo=20),
    testo("Quanti articoli servono", "Sezione 5.5", [
        "Per una testata nuova: quanti articoli bisogna raccogliere?",
        "Ogni testata valutata con **k articoli a caso** (1, 2, 3, 5, 10, 20, 50), 20 volte",
        "Con **1 articolo**: Mistral 17,3 contro 20,5 di TF-IDF (ρ 0,62 contro 0,39)",
        "Con **5 articoli** Mistral batte già TF-IDF che li legge tutti",
        "Dopo **~10 articoli** le curve sono piatte",
    ]),
    grafico("Quanti articoli servono", "Figura 6", "fig_curves"),
    testo("Orientamento politico", "Sezione 5.7", [
        "Abbiamo controllato se l'orientamento delle testate pesa sugli errori",
        "Orientamento e voto vanno insieme: sinistra 80, destra 54 → si confronta **a parità di voto**",
        "I modelli ordinano le testate anche **dentro** ogni orientamento",
        "A parità di voto: **~9 punti** in più alle testate di sinistra (tre controlli diversi, 8,7–9,6)",
        "Lo presentiamo come **cautela** per l'uso",
    ]),
    grafico("Orientamento politico", "Figura 7", "fig_errors"),
    grafico("Il tool", "Nel paper (Appendice M), non un demo paper", "fig_tool"),
    grafico("Il tool", "La pagina dei modelli: quelli del paper (uno per fold) e quelli personalizzati", "tool_models"),
    # ------------------------------------------------------------------ appendici
    testo("Le appendici", "L'ex supplementary è dentro il PDF, dopo i ringraziamenti", [
        "Appendici **A–N**",
        "Per ognuna: la **critica** a cui risponde, **come** è fatta, il **risultato**",
    ]),
    appendice("A", "More metrics", "«Solo MAE e Spearman?»",
              ["R², RMSE e AUC (affidabile sì/no) sulle stesse previsioni"],
              "stesso ordine dei modelli; AUC 0,88 contro 0,86 di TF-IDF", "tab_more"),
    appendice("B, E, F", "Test statistici, dettagli, prompt", "«Tanti confronti, falsi positivi? È riproducibile?»",
              ["Holm in tre famiglie di test", "tutti gli iperparametri", "il prompt del zero-shot"],
              "i confronti citati nel testo, con intervalli e p corretti"),
    appendice("C", "L'anonimizzazione in dettaglio", "«L'anonimizzazione funziona? Che cosa resta?»",
              ["le impostazioni di GLiNER e quante sostituzioni fa ogni passo", "il nome della testata rimasto"],
              "il nome resta nell'1,6% degli articoli (prima 37,8%)", "fig_anonymization"),
    appendice("D", "Chi resta fuori", "«Il campione è rappresentativo?»",
              ["le testate perse a ogni passo della pulizia: voto, livello, orientamento"],
              "le non raccolte sono molto meno affidabili (voto medio 41 contro 69)", "tab_selection"),
    appendice("G", "Controlli di robustezza e «solo il nome»", "«Il modello conosce le testate dal pretraining?»",
              ["un vettore dal solo nome di dominio, stessa ridge e stessi fold"],
              "dal solo nome ρ 0,35–0,39, come l'orientamento; dal testo 0,74"),
    appendice("H", "Altri split delle testate", "«Il risultato dipende dallo split?»",
              ["altri 4 split fatti allo stesso modo, seed diversi"],
              "Mistral batte TF-IDF di 2,0–2,3 punti in ogni split", "tab_splits"),
    appendice("I", "Testate sorelle", "«Le sorelle sono davvero simili?»",
              ["copie prima e dopo la deduplicazione", "voti delle sorelle", "similarità e guadagno"],
              "23 testate su 31 hanno lo stesso voto di tutte le sorelle; più testo simile, più cala l'errore"),
    appendice("J", "Anonimizzazione a regole", "«La scelta dell'anonimizzazione cambia i risultati?»",
              ["tutta la pipeline rifatta con 6 regole al posto di GLiNER"],
              "i modelli linguistici cambiano di meno di 0,25 punti", "tab_anon"),
    appendice("K", "Boilerplate", "«I modelli leggono le righe fisse del sito?»",
              ["TF-IDF riaddestrato senza quelle righe", "Mistral scomposto token per token"],
              "il boilerplate pesa ~2 punti; il voto viene dal corpo degli articoli", "tab_reading"),
    appendice("L, N", "Errori in dettaglio e cosa altro abbiamo provato", "«Il design è adattato a queste testate?»",
              ["errori per livello e orientamento modello per modello", "l'elenco dei tentativi precedenti"],
              "risultati stabili sui 5 split; non è una replica su testate nuove (lo diciamo)"),
]

pagine = [q for p in pagine for q in (p if isinstance(p, list) else [p])]

# ------------------------------------------------------------------ scrittura (solo le immagini usate)
z = zipfile.ZipFile(MODELLO)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
import re  # noqa: E402

manifest = re.sub(r' ?<manifest:file-entry manifest:full-path="(?:Pictures|Thumbnails)/[^"]*"[^>]*/>\n?', "", manifest)
corpo = "".join(re.sub(r'draw:name="[^"]*"', f'draw:name="page{n}"', p, count=1) for n, p in enumerate(pagine, 1))
inizio, fine = c.find("<draw:page "), c.rfind("</draw:page>") + len("</draw:page>")
c = c[:inizio] + corpo + c[fine:]
for nome in S.immagini:
    manifest = manifest.replace("</manifest:manifest>",
                                f' <manifest:file-entry manifest:full-path="{nome}" manifest:media-type="image/png"/>\n</manifest:manifest>')
with zipfile.ZipFile(OUT, "w") as zo:
    zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
    for info in z.infolist():
        if info.filename == "mimetype" or info.filename.startswith("Pictures/") or info.filename.startswith("Thumbnails/"):
            continue
        dati = {"content.xml": c.encode(), "META-INF/manifest.xml": manifest.encode()}.get(info.filename) or z.read(info.filename)
        zo.writestr(info, dati, compress_type=zipfile.ZIP_DEFLATED)
    for nome, percorso in S.immagini.items():
        zo.write(percorso, arcname=nome)
print(f"{OUT.name}: {len(pagine)} slide, {len(S.immagini)} immagini")
