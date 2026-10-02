#!/bin/bash
# Compila i supplementary e li ottimizza per il visualizzatore web di Anonymous GitHub.
#
# Il viewer di Anonymous GitHub non renderizza NESSUNA immagine dentro un PDF:
# ne' le figure vettoriali del backend PGF, ne' PNG, ne' JPEG. Le figure sono
# quindi riprodotte come immagini nel README del repo, dove il sito le mostra
# inline; qui restano vettoriali, che e' la resa migliore per chi scarica il PDF.
#
# Il PDF grezzo di xdvipdfmx pesa ~1.2 MB e non e' linearizzato. Ghostscript lo
# riscrive (~410 KB) e qpdf lo linearizza, cosi' si apre piu' in fretta.
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
tectonic -X compile supplementary.tex --reruns 3
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/prepress \
   -dDetectDuplicateImages=true -dNOPAUSE -dBATCH -dQUIET \
   -sOutputFile=_gs.pdf supplementary.pdf
qpdf --linearize _gs.pdf _fix.pdf
mv _fix.pdf supplementary.pdf
rm -f _gs.pdf
echo "supplementary.pdf: $(pdfinfo supplementary.pdf | grep -oP 'Pages:\s+\K\d+') pagine, $(du -h supplementary.pdf | cut -f1), linearizzato"
