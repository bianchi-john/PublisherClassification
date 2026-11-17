# Piano di revisione del paper

Da due revisioni indipendenti della bozza (metodo e statistica; contributo e posizionamento), entrambe "revisione
maggiore", teniamo i punti che un revisore userebbe davvero contro il paper. Decisioni prese con John il 13/11/2026.
Il codice è nella pipeline (`Code/Pipeline`, passi 5, 7, 13, 14, 15), coda sul server `run_review.sh`.

Stato: ⬜ da fare · 🟡 in corso · ✅ fatto. Ultimo aggiornamento: 13/11/2026.

---

## Analisi

**1. Quanto conta il protocollo ("scala delle fughe")** ✅ analisi (passo 14) · ⬜ testo
- *Problema.* Diciamo che fughe di testata, sorelle, copie e identificativi gonfiano i risultati, ma non lo misuriamo.
- *Soluzione.* TF-IDF, Qwen3-8B e Mistral con la ridge del passo 7, testando sempre sulle **stesse** 226 testate e
  sugli stessi articoli: split per articolo; **sorelle** (le 89 testate dello stesso proprietario, con tutti i loro
  articoli e le copie, messe apposta in addestramento in ogni fold); L2 testo originale; L3 solo regole; L4 GLiNER (il
  paper). Le sorelle si leggono sulle 31 testate che ne hanno una, con la somiglianza tra test e addestramento (la
  parentela è solo un indizio: conta che i testi siano simili) e lo scarto del voto NewsGuard dentro il gruppo.
  L0/L1 sulle 226 tolti (decisione di John, 17/11): diluivano l'effetto. Risultati 17/11: vedi registro.

**2. Cosa del voto NewsGuard vede il testo** ⬜ solo testo
- *Decisione.* Nessuna analisi per criterio: anche i criteri apparentemente "testuali" richiedono verifiche esterne
  (altre fonti, pagine del sito, storico delle correzioni), e titoli e firme sono rimossi o mascherati. Lo spieghiamo
  e introduciamo la metafora del DNA: un capello non dice nulla a una persona, ma in laboratorio rivela **tratti**
  dell'individuo senza bisogno di sapere chi è (tratti, non identità). La curva per numero di articoli diventa "quanti
  capelli servono". In introduzione e discussione.

**3. Il divario politico** 🟡 passo 15
- *Problema.* Le testate di destra hanno voti medi più bassi: un predittore schiacciato verso il centro dà comunque, a
  parità di voto, valori più bassi al gruppo con la media più bassa (Kleinberg et al. 2017; Chouldechova 2017).
- *Soluzione.* Divario a parità di voto con retta, spline ed effetti fissi per livello; divario che produrrebbe la
  sola compressione; calibrazione per gruppo. Nessun altro esperimento politico.
- *Prima lettura (dati attuali).* Mistral: divario 8,7–9,6 punti; la sola compressione ne produrrebbe 12,6; a parità
  di previsione sinistra e destra hanno lo stesso voto vero (+0,8 [−7,4; +8,9]). Non è un pregiudizio del modello ma
  la conseguenza delle base rate; ModernBERT addestrato invece non è calibrato (+12,8).

**4. Le baseline che mancano** 🟡 passi 5, 7, 13
- LLM zero-shot (Mistral-Small-24B-Instruct, stessa famiglia del modello congelato).
- ModernBERT congelato + ridge (stesso modello di quello addestrato).
- Qwen3-Embedding-0.6B congelato + ridge: separa la dimensione (0,6B contro 8B, stessa istruzione), ed è anche il
  modello leggero per il sito (CPU, 8 GB di RAM).
- Lo strato di Mistral (27 o 40) ora si sceglie sui fold interni: esce il 27 in tutti i fold, i numeri di Mistral non cambiano.
- Qwen3-0.6B e ModernBERT congelato scelgono spesso la penalità più bassa della griglia (10); allargandola fino a 0,01 la ρ cambia meno di 0,001: basta una frase.

**5. Testate note e meno note** ❌ tolta (decisione di John, 17/11)
- *Soluzione.* Le 226 testate divise in due metà per rango Tranco (lista 947W2, 13/11/2026); il vantaggio sui modelli
  congelati in ciascuna metà, senza togliere dati. Per le date basta una frase: gli articoli sono dal 2025, Mistral-
  Small-24B-Base-2501 è di gennaio 2025.

## Testo

**6. Affermazioni più forti dei dati** ⬜
- Con la correzione di Holm (passo 15, già calcolato): "con 5 articoli Mistral ordina meglio di TF-IDF con tutti" **non
  è significativa** sul ρ (+0,038 [−0,005; 0,089]), lo è sul MAE (−1,7): l'abstract va corretto. 5 dei 31 confronti
  delle curve non sono più significativi.
- Aggregazione confrontata alla stessa unità: 1 articolo contro tutti, per testata (fatto nel passo 15).
- Spiegazioni sui meccanismi come ipotesi; compressione "in parte dal disegno"; "nessuna differenza significativa"
  invece di "invariato"; "nuove o piccole" e l'uso pratico come prospettiva.
- Baseline nuove: mediana (MAE 22,1 invece di 22,6) e solo orientamento (ρ 0,36, MAE 18,5, entro 20 il 64%).

**7. Dataset e limiti** 🟡 passo 14–15
- Chi resta fuori (tabella dal passo 15), "proprietario registrato da NewsGuard", fonte dell'orientamento (campo
  "Orientation" di NewsGuard), critiche di parzialità rivolte a NewsGuard.

**8. Lavori più vicini** ⬜
- Rashkin 2017, Baly 2018, Horne 2019, Bozarth & Budak 2020, Fairbanks 2018; confronto su: solo testo, editori
  separati, gruppi separati, anonimizzato, punteggio continuo. Bibliografia: niente "verify" stampati, doppione
  Przybyła & Soto unito.

**9. Tool più corto** ⬜
- Un paragrafo di disponibilità; il resto nel supplemento; correggere "not even indirectly".

**10. Incoerenze puntuali** ⬜
- 704 contro 797 sorelle; 81 contro 61; "fino a 200" contro "da 10 a 201"; "circa dieci punti" contro 6,7–9,6;
  "entro 20 ≈ un livello"; iperparametri di ModernBERT; grassetto di Qwen su "entro 20".

## Registro

| Data | Cosa |
|---|---|
| 12/11/2026 | Piano da due revisioni indipendenti, ridotto a 5 analisi e 5 correzioni di testo |
| 13/11/2026 | Decisioni di John: scala delle fughe con "solo nomi", criteri solo testo + DNA, notorietà per metà, niente esperimenti politici in più, Qwen3-0.6B anche per il sito. Codice scritto nella pipeline (passi 13–15); il passo 15 riproduce esattamente i numeri attuali del paper. GPU utilizzabile (solo `nvidia-smi` rotto) |
| 13/11/2026 | Coda partita alle 16:01 (passi 5 e 7 fatti in 20 minuti). Codice ripulito per leggibilità e provato sul server (risultati identici); resta in locale, si installa dopo la coda. Fine prevista sabato sera |
| 17/11/2026 | Coda finita (17/11 03:10), risultati identici alla copia di prova, scala completa: lo split per articolo gonfia ρ di 0,17–0,23 (e mette TF-IDF davanti a Qwen), sorelle e copie ±0,02, L1–L3 entro 0,013 da L4. Decisione di John: la misura principale è il MAE (Mistral − TF-IDF −2,0, significativo anche con Holm); ρ come misura secondaria |
| 17/11/2026 | Decisione di John: niente analisi della notorietà (punto 5) nel paper |
| 17/11/2026 | Sorelle rifatte (screen t20_sisters, 11:46). Sulle 31 testate con una sorella in addestramento il MAE scende di 11,6 (TF-IDF), 6,9 (Qwen), 8,7 (Mistral) punti, quasi come lo split per articolo; sulle 195 senza non migliora (+0,4…+0,9); sulle 226 −0,3…−0,8 (n.s., diluito). Somiglianza con l'addestramento 0,19 → 0,60, 50% degli articoli con una quasi-copia tra le sorelle; il guadagno segue la somiglianza (ρ 0,60 per TF-IDF, 0,25–0,32 per gli LLM, n.s.). Stesso proprietario: 74% stesso voto identico, scarto medio 3,8 contro 29,6 a caso |
