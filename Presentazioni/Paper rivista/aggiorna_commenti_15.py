"""Quindicesima revisione del 26 novembre di PaperRivista.odp (58 slide): l'appendice B finisce con quello che conta per
John: che cosa è significativo e cosa no (dopo Holm), e se il ρ e il τ di Mistral senza asterisco sono un problema per il
paper. Numeri: Dataset/tentativi/20_FromScratch/paper_main_table.csv e paper_curves.csv; split: Tabella 10 del paper.
Uso:
    python aggiorna_commenti_15.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zr"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 58, len(p)
assert "3. La risposta, in appendice B" in S.testo(p[30])


def B(sotto, voci):
    return S.pagina("Appendice B", sotto, voci, 20, alto_testo=11.6)


p[30] = S.togli_punto(p[30], "Esito:")           # l'esito va nelle due slide nuove
esito = [
    B("4. Che cosa è significativo e cosa no", [
        ("**Sì**, significativamente meglio di TF-IDF dopo la correzione:", [
            "l'**errore** (MAE), la misura principale: Mistral −2,0 e Qwen3-8B −1,5 con tutti gli articoli; Mistral "
            "−3,2 con 1 articolo e −2,2 con 5",
            "l'**ordine** (ρ) con **pochi articoli**: Mistral +0,22 con 1 articolo e +0,12 con 5 (Qwen3-8B +0,11 e "
            "+0,07)",
            "entro 20 punti: Qwen3-8B +8,4",
        ]),
        ("**No**:", [
            "l'**ordine** (ρ e τ) con **tutti** gli articoli: Mistral +0,06, Qwen3-8B +0,04",
            "entro 20 punti per Mistral (+6,6)",
            "ModernBERT e Qwen3-0.6B, con tutti gli articoli: nessuna differenza significativa da TF-IDF",
        ]),
    ]),
    B("5. È un problema per il paper?", [
        "La critica vera: «con tutti gli articoli il vostro modello non **ordina** le testate significativamente meglio "
        "di TF-IDF». Su questo ha ragione, e il paper lo dice",
        "Ma la misura **principale** del paper è l'**errore** (Sezione 4), e lì il vantaggio è significativo",
        "Con **pochi articoli**, il caso delle testate nuove, anche l'ordine è significativamente migliore, e di molto "
        "(+0,22 con 1 articolo)",
        "Con tutti gli articoli Mistral ordina meglio in **tutti e 5** gli split (+0,03 a +0,07): la differenza è "
        "piccola perché TF-IDF ordina già bene (0,67), e 226 testate non bastano per l'asterisco",
        "Il contributo centrale non dipende da questo: senza le protezioni del protocollo l'errore cambia di **6–12 "
        "punti**, molto più della differenza tra i modelli",
    ]),
]
nuove = p[:31] + esito + p[31:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 58)")
