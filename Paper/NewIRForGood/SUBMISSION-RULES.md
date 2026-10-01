# ECIR 2027 — Track "IR for Good": regole di submission e checklist di conformità

> Documento di riferimento per verificare velocemente che il paper rispetti le regole della
> track, senza ritornare ogni volta sul sito.
>
> **Fonte primaria:** <https://www.ecir2027.co.uk/call-for-ir-for-good-papers>
> **Verificato il:** 30 settembre 2026 — **aggiornato il:** 1 ottobre 2026
> Le frasi tra virgolette in inglese sono **citazioni letterali** dal sito ufficiale.
> Tutto ciò che è marcato *[interpretazione]* è una mia deduzione, non una citazione.

---

## 0. In sintesi — i vincoli che possono farci rigettare per motivi formali

| # | Vincolo | Valore |
|---|---------|--------|
| 1 | Deadline paper | **5 ottobre 2026, 23:59 AoE** |
| 2 | Lunghezza | **6–12 pagine** di contenuto + referenze illimitate |
| 3 | Cosa conta nel limite | Contenuto **+ appendici + sezione Theory of Change** |
| 4 | Template | **Springer LNCS** (LaTeX o Word) |
| 5 | Revisione | **Doppio cieco** |
| 6 | Sezione obbligatoria | **Theory of Change** (per paper che propongono metodi/strumenti) |
| 7 | Sistema | EasyChair — <https://easychair.org/conferences/?conf=ecir27> |
| 8 | Abstract anticipato | **NON richiesto** per questa track (a differenza dei Full Papers) |

---

## 1. Date

Citazione letterale dalla sezione *IR for Good Track Dates*:

> - Paper submission: October 5, 2026, 11:59pm (AoE)
> - PC reviewing deadline: 18 November 2026
> - SPC reviewing deadline: 30 November 2026
> - Paper notification: 7 December 2026
> - Camera-ready deadline: January 10, 2027

**Attenzione — differenza rilevante con la track Full Papers.** I Full Papers richiedono un
abstract entro il **21 settembre 2026**; la track IR for Good **non lo richiede**: esiste solo
la deadline del paper completo. Ho verificato esplicitamente questo punto perché la data
dell'abstract dei Full Papers è già passata. Per noi non è un problema.

La conferenza si tiene dal **21 al 25 marzo 2027** a Southampton (UK).

---

## 2. Lunghezza

Citazione letterale:

> "at least 6 pages (the sixth page should have at least some content but not necessary to
> fill it) and at most 12 pages in length plus additional pages for references"

E, in modo esplicito:

> "The 'Theory of Change' section and Appendices count toward the page limit"

**Conseguenze operative:**
- Le **referenze sono illimitate** e NON contano.
- **Appendici e Theory of Change contano**: non esiste la scappatoia "sposto in appendice".
  L'unico overflow gratuito è il **supplementary esterno** (OSF), che non fa parte del PDF.
- Criterio di valutazione dichiarato: i revisori giudicano anche se
  *"paper length is commensurate with its contribution"* — quindi non è obbligatorio arrivare
  a 12 pagine, ma la lunghezza deve essere proporzionata al contributo.

**Collocazione delle appendici** (se ne mettiamo):

> "Please put appendices before the references for paper submission"

Cioè: appendici **prima** della bibliografia, non dopo. È l'opposto della convenzione LaTeX
abituale — da ricordare se aggiungiamo un `\appendix`.

---

## 3. Template e formato

- Vanno seguite *"Springer's authors' guidelines"* usando *"their proceedings templates,
  either for LaTeX or for Word"*.
- Gli atti sono pubblicati nella *"Springer Lecture Notes in Computer Science series"*.
- *"All submissions must be written in English"*.
- Gli ORCID sono incoraggiati.

**Concretamente per noi** *[interpretazione, pratica standard LNCS]*:
- `\documentclass[runningheads]{llncs}` + `llncs.cls`
- `\bibliographystyle{splncs04}` + `splncs04.bst`
- Abstract e keyword obbligatori; in LNCS l'abstract va **dopo** `\maketitle`, le keyword
  vanno **dentro** l'ambiente abstract con `\keywords{... \and ...}`
- **Non alterare** margini, corpo del font o interlinea del template per guadagnare spazio:
  è la violazione formale più facile da individuare e la più banale da evitare.

> ⚠️ Le linee guida dettagliate di Springer sono dietro autenticazione
> (<https://link.springer.com/series/558/information-for-authors-and-editors>), quindi non ho
> potuto citarle alla lettera. I punti qui sopra sono prassi LNCS consolidata, non citazioni.

---

## 4. Doppio cieco

La revisione è:

> "double-blind peer review, with an initial first stage review followed by a second stage"

Indicazione esplicita del sito:

> "make some effort to avoid any issues that may harm the double-blindness of your submission"

### Preprint su arXiv — permesso, con cautele

Questo punto vale la pena citarlo per esteso, perché è quello che decide il nostro caso:

> "Submission is permitted for papers that have previously been made available as a technical
> report (e.g., in institutional archives or preprint archives like arXiv)."

> "However, we discourage this since it places anonymity at risk; in particular, please do not
> publish your paper at arXiv and submit to ECIR at the same time, some days before, or during
> the reviewing period of ECIR."

> "You might not want to use the exact same title and abstract for your ECIR submission (in
> case of acceptance at ECIR, the title of your submission still might be changed «back»)."

> "Please do not cite your technical report and make some effort to avoid any issues that may
> harm the double-blindness of your submission."

> "Reviewers will receive guidance that ask them to refrain from trying to break blindness if
> at all possible too, but be aware that the availability of a technical report for an ECIR
> submission can cause issues."

**Come ci riguarda.** La versione respinta da TCSS è su arXiv. Questo **non ci squalifica**:
il preprint è permesso, e lo scoraggiamento riguarda il pubblicare *durante* la finestra di
review, cosa che non stiamo facendo. Le tre precauzioni che il bando suggerisce sono state
applicate: titolo cambiato, abstract riscritto, technical report non citato.

**Checklist di anonimizzazione da applicare al PDF di submission** *[interpretazione operativa]*:

- [ ] Nomi degli autori rimossi dal front matter
- [ ] Affiliazioni (`\institute`) rimosse
- [ ] Indirizzi email rimossi
- [ ] **ORCID rimossi** — identificano univocamente gli autori. Vanno rimessi nel camera-ready
- [ ] **Ringraziamento SIMULARE rimosso** — il nome del progetto e l'ente finanziatore
      identificano il gruppo
- [x] **Supplementary**: ospitato su Anonymous GitHub
      (`anonymous.4open.science/r/supplementary-material-DD54/README.md`). Il repo GitHub
      sorgente è **privato** e il commit è firmato con identità neutra. Il PDF è anonimo di suo
      (verificato sui byte), perché Anonymous GitHub non anonimizza il contenuto dei PDF
- [x] **Modelli fine-tuned**: nessun link durante la review, nota "available upon acceptance".
      Il repo OSF è stato cancellato (compariva su Google insieme al preprint arXiv)
- [x] **Chiave BibTeX** neutra: era `bianchi2026supplementary` e finiva nel PDF come ancora
      hyperref `cite.bianchi2026supplementary`, invisibile nel testo ma presente nei byte
- [ ] Autocitazioni: controllare che non ci siano frasi tipo "in our previous work [12]" che
      collegano il paper agli autori. Riformulare in terza persona
- [ ] Metadati del PDF (autore/titolo) puliti
- [ ] Nessuna pubblicazione su arXiv prima o durante la revisione

---

## 5. Theory of Change — sezione OBBLIGATORIA

Richiesta per i paper che propongono *"new IR tools, methods, resources, and interventions"*.
*"Position papers and critiques are exempt from this requirement."*

**Il nostro paper propone un metodo → la sezione è obbligatoria.**

Deve rispondere a tre domande, citate letteralmente:

1. > "What is the identified societal need / problem, and how are the core contributions from
   > this current work expected to address them?"
2. > "What preconditions are necessary or what assumptions need to hold for this work to have
   > its desired effect"
3. > "What are possible negative externalities of this approach"

Attenzione al testo esatto della domanda 2: chiede anche *"and how likely are they to hold
true in practice?"*. Non basta elencare le assunzioni, va dichiarata la loro **plausibilità**.
La domanda 3 aggiunge *"is it plausible that this may lead to new or different harms?"*: il
punto non è solo che i danni siano grandi, ma che siano di **tipo nuovo**.

Dalla guidance dell'organizzatore della track:

> "This section should not be an afterthought; instead it should be a critical part of the key
> motivations for the work and as important as any other core sections."

> "it is not mandatory to name the section «Theory of change» but it should be apparent from
> the section title that it elaborates on how the work contributes towards desired societal
> outcomes."

> "Our goal is **not** to encourage authors to inflate their claims of societal impact but to
> rigorously deliberate on their sociotechnical assumptions."

**Collocazione.** Non è prescritta. Nel nostro paper la sezione sta **dopo gli Experiments e
prima di Discussion**, perché le esternalità sono *misurate* e non si possono argomentare
prima di aver mostrato i risultati; per non farla sembrare un'appendice, l'introduzione la
richiama subito dopo la lista dei contributi.

---

## 6. Scope della track

La track raccoglie ricerca IR "impactful, high-quality, and societally-motivated". Temi
dichiarati includono: bias algoritmico, fairness, trasparenza, interpretabilità,
spiegabilità, affidabilità, **misinformation e disinformation**, hate speech, replicabilità,
trasferibilità, robustezza, incertezza, sicurezza ed etica. Aree di intersezione citate:
accessibilità e disability justice, decolonizzazione e giustizia razziale, sostenibilità e
giustizia ambientale, democrazia, privacy, salute pubblica, equità.

**Il nostro fit:** misinformation / affidabilità delle fonti + analisi di bias politico.
È centrato. Da valorizzare esplicitamente nell'introduzione e nella Theory of Change.

---

## 7. Policy varie

**Doppia submission:**
> "Papers submitted to ECIR 2027 should be substantially different from papers that have been
> previously published, or accepted for publication, or that are under review at other venues."

Eccezioni ammesse: presentazioni a conferenze/workshop senza atti, e technical report su
preprint server. *Il nostro caso è a posto: il paper è stato **respinto** da IEEE TCSS, quindi
non è né pubblicato né sotto revisione altrove.*

**AI generativa:** si applica la *Springer AI policy*. La policy citata sul sito riguardo agli
strumenti AI è formulata per i **revisori** (*"AI-assisted tools may be used solely for language
editing... They should not be used to evaluate submissions"*) e vieta di caricare submission
confidenziali su servizi AI esterni. Per gli autori vale la policy Springer: gli LLM non
possono essere elencati come autori e l'uso va dichiarato dove richiesto. *[interpretazione]*

**Etica:** conformità obbligatoria alle policy ACM su authorship, codice etico, conflitti di
interesse, plagio/falsificazione e molestie, più la AI policy di Springer.

**Presentazione:** almeno un autore deve registrarsi e presentare il lavoro accettato.

---

## 7bis. Supplementary material — ECIR non lo prevede

Verificato su entrambe le pagine (IR for Good e Full Papers): **nessuna delle due menziona
supplementary material, repository esterni, OSF, Zenodo o upload aggiuntivi**. L'unica cosa
prevista sono le appendici, e per quelle vale:

> "The 'Theory of Change' section and Appendices count toward the page limit."
> "Please put appendices before the references for paper submission."

**Conseguenza pratica.** Un'appendice interna costa pagine che non abbiamo. Il supplementary
è quindi un documento **autonomo ospitato fuori** (`supplementary.tex` → Anonymous GitHub) e
citato dal paper, come faceva la versione TCSS. Non è un modo per aggirare il limite: dentro
ci sono solo matrici di confusione complete e curve di convergenza, cioè materiale
illustrativo, non argomenti del paper.

**Attenzione per il camera-ready.** Zenodo non permette di sostituire i file di un record già
pubblicato (*"Files in the record however can only be edited (added, modified or deleted)
after publication by contacting support"*) e la cancellazione è possibile solo entro 30 giorni.
Il record TCSS esistente va quindi lasciato com'è; alla camera-ready si crea una **nuova
versione**.

---

## 8. Checklist di conformità — stato al 1 ottobre 2026

| # | Requisito | Stato | Note |
|---|-----------|:-----:|------|
| 1 | Template LNCS (`llncs.cls` + `splncs04.bst`) | ✅ | Convertito da IEEEtran; compila pulito |
| 2 | Lingua inglese | ✅ | |
| 3 | Abstract + keyword nel formato LNCS | ✅ | `\keywords` dentro l'abstract |
| 4 | **Lunghezza ≤ 12 pagine di contenuto** | ❌ | **22 pagine di contenuto** (25 col bibliography). Da tagliare ~10 pp. È l'unico bloccante rimasto |
| 5 | Lunghezza ≥ 6 pagine | ✅ | |
| 6 | **Sezione Theory of Change** | ✅ | Scritta, pp. 19–21. Copre le tre domande, con giudizio di plausibilità per ogni precondizione e danni qualificati come di tipo nuovo. Richiamata dall'introduzione |
| 7 | **Anonimizzazione doppio cieco** | ✅ | Interruttore `\anonymoustrue` in `main.tex` e `supplementary.tex`. Verificato sui byte del PDF: nessun nome, affiliazione, ORCID, SIMULARE, OSF o Zenodo. Titolo e abstract cambiati rispetto al preprint arXiv (0 n-grammi di 6+ parole in comune). Chiave BibTeX rinominata (era `bianchi2026supplementary`, leak via ancore hyperref). Modelli non linkati durante la review. |
| 8 | Appendici prima delle referenze | ➖ | Al momento non ci sono appendici |
| 9 | Nessuna doppia submission | ✅ | Respinto da TCSS, non sotto revisione altrove |
| 10 | Preprint arXiv preesistente | ✅ | Permesso dal bando. Applicate le tre cautele suggerite: titolo cambiato, abstract riscritto (0 n-grammi di 6+ parole in comune col preprint), technical report non citato. Non pubblicare nulla di nuovo su arXiv fino a dicembre 2026 |
| 11 | ORCID (per il camera-ready) | ✅ | Nel ramo `\else` dell'interruttore; tornano commentando `\anonymoustrue` |
| 12 | Titolo diverso dal preprint | ✅ | *Source-Level Reliability Estimation under Publisher-Disjoint Evaluation*. Da riportare al titolo originale in caso di accettazione, se si vuole |
| 13 | Submission su EasyChair `ecir27` | ⏳ | Da fare entro il 5 ottobre 2026, 23:59 AoE |

Legenda: ✅ conforme · ❌ non conforme · ⏳ da fare · ➖ non applicabile

---

## 9. Procedura di verifica

**Step 1 — check automatico del formato.** Dalla cartella `Paper/NewIRForGood/`:

```bash
tectonic -X compile main.tex --keep-intermediates --reruns 3
```

Per misurare le **pagine di contenuto** (escluse le referenze), inserire temporaneamente
`\label{ENDOFCONTENT}` subito prima di `\begin{credits}`, ricompilare e leggere:

```bash
grep ENDOFCONTENT main.aux     # il secondo numero è la pagina
```

**Step 2 — check automatico dell'anonimato.** Non fidarsi del testo estratto: i leak più
insidiosi stanno nei metadati del PDF e negli stream compressi (ancore hyperref, percorsi di
sistema). Questo controlla entrambi:

```python
from pypdf import PdfReader
import re, zlib
raw = open("main.pdf", "rb").read()
blobs = [raw]
for st in re.findall(rb"stream\r?\n(.*?)endstream", raw, re.S):
    try: blobs.append(zlib.decompress(st))
    except Exception: pass
hay = b"\n".join(blobs)
for t in [b"Bianchi", b"Pratelli", b"Pinelli", b"Petrocchi", b"imtlucca", b"iit.cnr",
          b"SIMULARE", b"IMT School", b"osf.io", b"zenodo", b"/home/john",
          b"From Articles to Publishers"]:
    n = len(re.findall(re.escape(t), hay, re.I))
    if n: print(f"LEAK {t.decode()}: {n}")
print((PdfReader("main.pdf").metadata or {}).get("/Author"), 
      (PdfReader("main.pdf").metadata or {}).get("/Title"))   # devono essere vuoti
```

Atteso: nessun LEAK, metadati `/Author` e `/Title` vuoti. Fanno eccezione **Pratelli** e
**Pinelli**, che compaiono legittimamente in bibliografia (due auto-citazioni in terza persona,
che il doppio cieco permette e anzi si aspetta, più Pinelli come *editor* degli atti ECIR 2025
nel riferimento a TROPIC).

**Step 3 — distanza dal preprint.** L'abstract non deve condividere sequenze lunghe di parole
con la versione su arXiv, perché è così che i motori di ricerca fanno match:

```python
def ng(w, n): return {" ".join(w[i:i+n]) for i in range(len(w)-n+1)}
# atteso: 0 n-grammi condivisi per n >= 5
```

**Step 4 — check manuale contro la tabella della §8.** Da rifare a ogni revisione
significativa e, obbligatoriamente, subito prima di caricare su EasyChair.

---

## 10. Fonti

- [Call for IR for Good Papers — ECIR 2027](https://www.ecir2027.co.uk/call-for-ir-for-good-papers) (fonte principale)
- [Call for Full Papers — ECIR 2027](https://www.ecir2027.co.uk/call-for-full-papers) (per confronto su date e policy comuni)
- [Calls — ECIR 2027](https://www.ecir2027.co.uk/calls) (policy etiche trasversali)
- [EasyChair ECIR 2027](https://easychair.org/conferences/?conf=ecir27)
- [Springer LNCS — information for authors](https://link.springer.com/series/558/information-for-authors-and-editors) *(non accessibile senza autenticazione: i dettagli LNCS in §3 non sono citazioni letterali)*
- [What is IR-for-Good? — Bhaskar Mitra](https://bhaskar-mitra.github.io/posts/2025/09/01/what-is-ir-for-good/) (guidance dell'organizzatore della track sulla Theory of Change, §5)
- [Zenodo — Manage records](https://help.zenodo.org/docs/deposit/manage-records/) (immutabilità dei file dopo la pubblicazione, §7bis)
- [Anonymous GitHub](https://github.com/tdurieux/anonymous_github) (hosting anonimo del supplementary, §4)
