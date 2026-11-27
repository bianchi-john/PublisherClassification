"""Venticinquesima revisione (27 novembre) di PaperRivista.odp (69 slide): l'appendice F come la B, a elenchi di poche
parole, una slide per idea, con all'inizio a che cosa serve lo zero-shot (la prova «chiedere direttamente a un modello
linguistico»). La slide con il testo del prompt resta quella di prima.
Uso:
    python aggiorna_commenti_25.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "yb"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 69, len(p)
f = [i for i, x in enumerate(p) if "Appendice F" in S.titolo_di(x)]
assert f == [43, 44, 45, 46, 47], f
assert "Il prompt: il testo" in S.testo(p[44])


def F(sotto, voci):
    return S.pagina("Appendice F", sotto, voci, 24, alto_testo=11.6)


appendice_f = [
    F("A che cosa serve lo zero-shot", [
        "La domanda di tutti: «perché non chiedere **direttamente** a un modello linguistico?»",
        "Lo zero-shot è **questa prova**",
        "Nessun addestramento: il modello legge e risponde",
        "È un **riferimento**, non uno dei nostri modelli",
    ]),
    F("Quale modello", [
        "Mistral-Small-24B-**Instruct**: la versione che risponde alle domande",
        "Stessa famiglia del Mistral dei nostri vettori",
        "**Solo lui** riceve il prompt",
        "Gli altri modelli imparano dai **voti** di training",
    ]),
    F("Come lo usiamo", [
        "Un articolo alla volta: **25.524** articoli",
        "Risponde con **un numero** da 0 a 100",
        "Voto della testata: la **media** delle risposte",
    ]),
    p[44],
    F("Che cosa gli chiediamo", [
        "**Contesto**: che cos'è NewsGuard, i 9 criteri",
        "**Avviso**: i nomi sono nascosti ([OUTLET], [AUTHOR], …)",
        "**Domanda**: il voto del sito, un solo numero",
    ]),
    F("La critica", [
        "«Il prompt è scritto **male**»",
        "«Con un prompt migliore andrebbe **meglio**»",
    ]),
    F("La risposta", [
        "Vero: ne abbiamo provato **uno solo**",
        "Ma è solo un **riferimento**",
        "Le conclusioni del paper **non dipendono** da lui",
    ]),
    F("Il risultato", [
        "Zero-shot: errore **24,4**, peggio della costante (22,1)",
        "Tutti i modelli addestrati: **15,8–18,8**",
        "Lo stesso Mistral con una parte addestrata sui voti: il **migliore**",
        "Senza addestramento non sa che numero dare: quasi sempre **65 o 45**",
    ]),
]
nuove = p[:43] + appendice_f + p[48:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 69)")
