#!/bin/bash
# Compila i supplementary e li rende visualizzabili nel viewer di Anonymous GitHub.
#
# IL PROBLEMA. Quel viewer disegna il testo e i tracciati vettoriali che stanno
# nel flusso di contenuto della pagina (le righe delle tabelle si vedono), ma
# ignora silenziosamente gli XObject: sia i Form XObject, in cui \includegraphics
# incapsula una figura PDF, sia gli Image XObject. Niente errori in console,
# semplicemente non li disegna. Rasterizzare le figure non serve: PNG e JPEG
# sono anch'essi Image XObject e restano invisibili.
#
# LA SOLUZIONE. Un giro attraverso PostScript, che gli XObject non ce li ha:
# il contenuto delle figure viene riversato direttamente nel flusso di pagina.
# Si usa pdftops di poppler e non ps2write di Ghostscript, perche' quest'ultimo
# converte i font in Type 3. Restano 5 Image XObject, le sfumature delle
# colorbar, che matplotlib emette come raster: quelle nel viewer non si vedranno.
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
tectonic -X compile supplementary.tex --reruns 3
pdftops -level3 supplementary.pdf _supp.ps
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/printer \
   -dNOPAUSE -dBATCH -dQUIET -sOutputFile=_supp.pdf _supp.ps
qpdf --linearize _supp.pdf supplementary.pdf
rm -f _supp.ps _supp.pdf
forms=$(python3 -c "import re;print(len(re.findall(rb'/Subtype\s*/Form',open('supplementary.pdf','rb').read())))")
echo "supplementary.pdf: $(pdfinfo supplementary.pdf | grep -oP 'Pages:\s+\K\d+') pagine, $(du -h supplementary.pdf | cut -f1), $forms Form XObject (devono essere 0)"
