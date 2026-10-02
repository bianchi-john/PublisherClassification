#!/bin/bash
# Compila i supplementary e li converte in PDF/A-1, che e' l'unica forma che il
# visualizzatore web di Anonymous GitHub renderizza per intero, figure comprese.
#
# COME CI SIAMO ARRIVATI. Quel viewer non disegnava le figure in nessuna delle
# forme provate: vettoriali incapsulate in Form XObject, rasterizzate in PNG,
# rasterizzate in JPEG, appiattite via PostScript. Sono state messe in repo 17
# varianti dello stesso documento, ciascuna costruita con una tecnica diversa, e
# l'unica che si vede correttamente e' la conversione in PDF/A-1.
#
# COSA FA LA CONVERSIONE. PDF/A-1 vieta la trasparenza, quindi Ghostscript
# appiattisce le pagine che contengono figure rasterizzandole a 720 DPI: alla
# vista sono indistinguibili dal vettoriale a qualunque zoom pratico. Le pagine
# di solo testo e tabelle (1, 6, 7) restano vettoriali e ricercabili; quelle con
# figure (2, 3, 4, 5, 8) no. E' il prezzo, ed e' accettabile.
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
tectonic -X compile supplementary.tex --reruns 3
gs -dPDFA=1 -dPDFACompatibilityPolicy=1 -sDEVICE=pdfwrite \
   -dBATCH -dNOPAUSE -dQUIET -sOutputFile=_supp.pdf supplementary.pdf
mv _supp.pdf supplementary.pdf
echo "supplementary.pdf: $(pdfinfo supplementary.pdf | grep -oP 'Pages:\s+\K\d+') pagine, $(du -h supplementary.pdf | cut -f1), PDF/A-1"
