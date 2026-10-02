#!/bin/bash
# Compila i supplementary e li rende visualizzabili nel viewer di Anonymous GitHub.
#
# IL PROBLEMA. Quel viewer disegna il testo e i tracciati vettoriali che stanno nel
# flusso di contenuto della pagina (le righe delle tabelle si vedono), ma ignora in
# silenzio gli XObject: sia i Form XObject, in cui \includegraphics incapsula ogni
# figura PDF, sia gli Image XObject. Nessun errore in console, semplicemente non li
# disegna. Rasterizzare le figure non serve: PNG e JPEG sono anch'essi Image XObject.
#
# LA SOLUZIONE. Un giro attraverso PostScript, che gli XObject non li prevede: il
# contenuto delle figure finisce direttamente nel flusso di pagina.
#
# PERCHE' PROPRIO COSI'. Due varianti sono state provate e scartate:
#   - pdftops di poppler: perde le codifiche dei sottoinsiemi Type 1 incorporati
#     nelle figure e sostituisce i font. La cifra 1 veniva disegnata come una "l",
#     quindi 411 si leggeva 4ll. Inaccettabile.
#   - ps2write a risoluzione di default: i glifi Type 3 escono come bitmap a bassa
#     risoluzione e il testo appare sgranato appena si ingrandisce.
# Con -r1200 i glifi Type 3 sono indistinguibili dall'originale a qualsiasi zoom.
#
# IL COSTO. Il giro in PostScript distrugge il livello di testo: il PDF non e' piu'
# ricercabile ne' copiabile. Restano 5 Image XObject, le sfumature delle colorbar,
# che nel viewer non si vedranno.
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
tectonic -X compile supplementary.tex --reruns 3
gs -sDEVICE=ps2write -r1200 -dNOPAUSE -dBATCH -dQUIET -sOutputFile=_supp.ps supplementary.pdf
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/printer \
   -dNOPAUSE -dBATCH -dQUIET -sOutputFile=_supp.pdf _supp.ps
qpdf --linearize _supp.pdf supplementary.pdf
rm -f _supp.ps _supp.pdf
forms=$(python3 -c "import re;print(len(re.findall(rb'/Subtype\s*/Form',open('supplementary.pdf','rb').read())))")
echo "supplementary.pdf: $(pdfinfo supplementary.pdf | grep -oP 'Pages:\s+\K\d+') pagine, $(du -h supplementary.pdf | cut -f1), $forms Form XObject (devono essere 0)"
