"""Diciannovesima revisione del 26 novembre di PaperRivista.odp (63 slide): l'appendice F (il prompt dello zero-shot) come
nel paper: dove e quando lo usiamo (solo Mistral-Small-24B-Instruct, solo per lo zero-shot), com'è fatto e che cosa
genera, la critica e la risposta (85% delle risposte 65 o 45: manca la scala, non il compito; la stessa famiglia con
una testa lineare è la migliore).
Uso:
    python aggiorna_commenti_19.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zv"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 63, len(p)
f = [i for i, x in enumerate(p) if "Appendice F" in S.titolo_di(x)]
assert f == [39, 40, 41], f


def F(sotto, voci):
    return S.pagina("Appendice F", sotto, voci, 20, alto_testo=11.6)


appendice_f = [
    F("Il prompt: dove e quando lo usiamo", [
        "In **un solo** modello: Mistral-Small-24B-**Instruct**, la versione che risponde alle domande del Mistral di cui "
        "usiamo i vettori",
        "Solo per lo **zero-shot** (passo 13 della pipeline): nessun addestramento, il modello legge e risponde",
        "Una volta per ognuno dei **25.524** articoli anonimi; il voto della testata è la **media** delle risposte, come "
        "per gli altri modelli",
        "Gli altri modelli non ricevono prompt: imparano il voto dalle testate di training (Qwen3 riceve solo una frase "
        "che dice che cosa deve rappresentare il vettore)",
    ]),
    F("Il prompt: com'è fatto e che cosa genera", [
        "**Messaggio di sistema**: «sei un analista dei media»; che cos'è NewsGuard; i **nove criteri** con i loro pesi "
        "(22 punti non pubblicare falsità, 18 informare in modo responsabile, …); sotto 60 la testata non è credibile",
        "**Messaggio utente**: che cosa sono i segnaposto [OUTLET], [AUTHOR], [URL], [EMAIL]; «stima il voto NewsGuard "
        "(0–100) del sito che ha pubblicato questo articolo, rispondi con un solo numero»; poi l'articolo",
        "**Genera** un numero da 0 a 100, sempre lo stesso per lo stesso articolo (al massimo 6 token); 24 risposte su "
        "25.524 senza numero prendono la media del training",
        "Il testo intero è nell'appendice F",
    ]),
    F("Il prompt: la critica", [
        "«Lo zero-shot va male perché il prompt è scritto male: con un prompt migliore un modello linguistico farebbe "
        "di più»",
    ]),
    F("Il prompt: la risposta", [
        "Il prompt dà al modello il compito **come lo definisce NewsGuard**: i nove criteri con i pesi, la soglia di 60, "
        "i segnaposto",
        "Quello che manca è la **scala**: l'85% delle risposte è **65 o 45**, il modello divide le testate in due "
        "gruppi ma non sa dove metterle tra 0 e 100",
        "Un modello della **stessa famiglia**, sugli stessi articoli, con una testa lineare addestrata sui voti di "
        "training, è il **migliore** di tutti: MAE **15,8** contro **24,4**",
        "Quindi l'informazione nel modello c'è: allo zero-shot mancano i **voti** da cui imparare la scala",
        "Lo zero-shot resta il riferimento per «chiedere direttamente» a un modello linguistico: per questo compito non "
        "basta",
    ]),
]
nuove = p[:39] + appendice_f + p[42:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 63)")
