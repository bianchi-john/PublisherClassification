"""Seconda revisione del 26 novembre di PaperRivista.odp: risolve i commenti che John ha lasciato sulla versione
precedente (64 slide). Lavora sul file di John e tiene le sue modifiche (le spiegazioni che ha tolto restano tolte):
- Figura 2 con i gruppi più piccoli, Figura 8 senza didascalia e con le proporzioni giuste, Figura 5 come confronto
  tra le distribuzioni dei voti veri e previsti;
- anonimizzazione in due passi: regole (link, e-mail, nome della testata), poi GLiNER (tutte le testate e i giornalisti);
- «testate non valutate» → «testate non raccolte»: che cosa sono, a quale critica rispondono;
- il protocollo: una slide sul problema e sull'approccio a passi, poi la Tabella 7 con le spunte.
Prima: compilare il paper, python figure/schermate.py e le figure «solo_*» senza didascalia. Uso:
    python aggiorna_commenti_2.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent / "Presentazioni precedenti" / "Stato finale"))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zd"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 64, len(p)


def titolo(i, atteso):
    assert atteso in S.titolo_di(p[i]), (i, S.titolo_di(p[i]))
    return p[i]


def solo_tabella(nome):
    """La schermata di una tabella senza didascalia: dal primo filetto (una riga quasi tutta scura) in giù."""
    im = Image.open(SCH / f"{nome}.png").convert("L")
    w, h = im.size
    px = im.load()
    riga = next(y for y in range(h) if sum(px[x, y] < 128 for x in range(w)) > 0.85 * w)
    uscita = SCH / f"solo_{nome}.png"
    im.crop((0, riga - 4, w, h)).save(uscita)
    return uscita


def con_figura(tit, sotto, img, y=3.4):
    """Solo titolo, sottotitolo e la figura, la più grande possibile."""
    return S.pagina(tit, sotto, [], extra=S.figura(img, y, largo_max=26.0, basso=15.5))


def figura_e_testo(tit, sotto, img, voci, corpo=20, alto=11.6, largo_testo=9.6):
    """La figura a sinistra, alta quasi quanto la slide; il testo a destra."""
    w, h = Image.open(img).size
    largo = alto * w / h
    extra = S.immagine(img, 0.8, 3.0, alto=alto)
    pag = S.pagina(tit, sotto, voci, corpo, alto_testo=10.0, extra=extra)
    vecchio = 'svg:width="25.199cm" svg:height="10.00cm" svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline"'
    assert pag.count(vecchio) == 1
    x = 0.8 + largo + 0.5
    return pag.replace(vecchio, f'svg:width="{min(largo_testo, 27.6 - x):.2f}cm" svg:height="10.00cm" svg:x="{x:.2f}cm" '
                                f'svg:y="4.0cm" presentation:class="outline"')


def rinomina(pag, vecchio, nuovo):
    assert pag.count(vecchio) == 1, vecchio
    return pag.replace(vecchio, nuovo)


nuove = [titolo(0, "Contributi del paper (1/2)"), titolo(1, "Contributi del paper (2/2)"), titolo(2, "Tabella 1"),
         titolo(3, "Il dataset")]

# ------------------------------------------------------------------ Figura 2: anche i gruppi più piccoli
titolo(4, "Gruppi di proprietà")
nuove.append(con_figura("Gruppi di proprietà", "Figura 2: una mappa per passo di pulizia", SCH / "solo_group_maps.png"))

# ------------------------------------------------------------------ anonimizzazione: regole, poi GLiNER
titolo(5, "Anonimizzazione")
nuove.append(S.pagina("Anonimizzazione", "Due passi: prima le regole, poi GLiNER", [
    "**Regole** (espressioni regolari): link, e-mail e **nome della testata** stessa → **[URL]**, **[EMAIL]**, "
    "**[OUTLET]**",
    "**GLiNER** (NER zero-shot, etichette «news outlet» e «journalist»): in più **tutte** le testate e **tutti** i "
    "giornalisti → **[OUTLET]**, **[AUTHOR]**",
    "Le regole tolgono di sicuro la traccia più diretta, il nome; GLiNER estende la pulizia a tutti i nomi",
    "Nel controtest **«rules only»** (Tabella 7) usiamo **solo le regole**",
], 24))
titolo(6, "Anonimizzazione")
nuove.append(con_figura("Anonimizzazione", "Figura messa in appendice/supplementary", SCH / "solo_anonymization.png", y=4.2))
nuove += [titolo(7, "Il dataset finale"), titolo(8, "La pipeline"), titolo(9, "Setup sperimentale"),
          titolo(10, "Risultati: articoli"), titolo(11, "Risultati: testate")]

# ------------------------------------------------------------------ Figura 5: le due distribuzioni
titolo(12, "Voto previsto e voto vero")
nuove.append(figura_e_testo("Voto previsto e voto vero", "Figura 5: le due distribuzioni", SCH / "solo_distributions.png", [
    "**Grigio**: voto NewsGuard; **colore**: voto previsto (media sugli articoli della testata)",
    "I previsti si **ammassano al centro**: i modelli comprimono la scala",
    "**22** testate hanno voto sotto 30; quasi nessuna previsione ci arriva",
]))
nuove.append(titolo(13, "Contro TF-IDF"))

# ------------------------------------------------------------------ testate non raccolte (non «non valutate»)
for i, n in zip(range(14, 19), range(1, 6)):
    titolo(i, f"Testate non valutate ({n}/5)")
nuove += [
    S.pagina("Testate non raccolte (1/5)", "Che cosa sono", [
        "Ottobre 2026: abbiamo provato a scaricare **tutte** le 1.293 testate politiche **valutate** da NewsGuard",
        "**619** non hanno dato articoli (sito offline, robot bloccati, nessuna pagina di articolo): sono le **non "
        "raccolte**",
        "Hanno il **voto** NewsGuard ma **non il testo**: il modello non può girare su di loro",
        "Sono molto meno affidabili: voto medio **41** contro **69**; nel livello 0–39 il **43%**, contro il **15%** "
        "del dataset",
    ], 20, alto_testo=11.6),
    S.pagina("Testate non raccolte (2/5)", "A quale critica rispondono", [
        "«Il dataset è **rappresentativo**? L'errore che riportate vale anche fuori dalle 226 testate?»",
        "Il modello serve per testate **non ancora valutate**: lì non possiamo misurare l'errore, perché **manca il voto**",
        "Le non raccolte sono una popolazione **meno affidabile** di cui conosciamo i voti",
        "Mistral sbaglia di **31** punti nel livello 0–39 e di **8,5** nel 60–74: con più testate del livello basso "
        "l'errore **sale**",
        "Domanda: quale sarebbe l'errore su una popolazione **come le non raccolte**? Usiamo solo i loro **voti**",
    ], 20, alto_testo=11.6),
    rinomina(p[16], "Testate non valutate (3/5)", "Testate non raccolte (3/5)"),
    rinomina(p[17], "Testate non valutate (4/5)", "Testate non raccolte (4/5)"),
    rinomina(p[18], "Testate non valutate (5/5)", "Testate non raccolte (5/5)"),
]

# ------------------------------------------------------------------ il protocollo: il problema, poi la tabella a spunte
for i, n in zip(range(19, 25), range(1, 7)):
    titolo(i, f"Quanto conta il protocollo ({n}/6)")
nuove += [
    S.pagina("Quanto conta il protocollo (1/3)", "Il problema e l'approccio a passi (Sezione 5.4)", [
        "Il protocollo blocca tre modi di **riconoscere** una testata invece di leggerla: la stessa testata in training, "
        "le **sorelle** in training, i **nomi** nel testo",
        "Il problema: non si sa **quanto** ciascuna protezione aiuti o meno i modelli",
        ("Allora le togliamo **una alla volta**, sempre sugli stessi articoli e fold di test:", [
            "**stesse testate** in training e in test (article split: articoli diversi delle stesse testate)",
            "**sorelle** in training: le 89 testate tolte dalla pulizia, con tutti i loro articoli",
            "**anonimizzazione**: nessuna (original text), solo **regole** (rules only), regole + **GLiNER** "
            "(protocollo completo)",
        ]),
    ], 20, alto_testo=11.6),
    S.con_grafico("Quanto conta il protocollo (2/3)", "Tabella 7: ✓ = sì, × = no", [
        "Article split: niente sorelle, perché le testate di test sono **già** in training",
    ], solo_tabella("tab_ladder"), 20, largo_max=26.0),
    rinomina(p[24], "Quanto conta il protocollo (6/6)", "Quanto conta il protocollo (3/3)"),
]

# ------------------------------------------------------------------ il resto, com'è; nell'appendice J si distinguono le due «regole»
resto = p[25:]
j = 56 - 25
assert "6 regole al posto" in S.testo(resto[j])
resto[j] = S.riscrivi(resto[j], "6 regole al posto", "Rifatta tutta la pipeline (anche ModernBERT) con una **versione "
                      "precedente**: 6 regole e niente GLiNER (non le «regole» della Tabella 7)")
nuove += resto

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ora ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 64), {len(S.immagini)} immagini nuove")
