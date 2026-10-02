#!/bin/bash
# Compila i supplementary. Nessuna post-elaborazione: il PDF che esce da tectonic
# e' gia' quello giusto.
#
# IL PROBLEMA. Il visualizzatore web di Anonymous GitHub disegna il testo e i
# tracciati vettoriali che stanno nel flusso di contenuto della pagina, ma ignora
# in silenzio gli XObject: sia i Form XObject, in cui \includegraphics incapsula
# ogni figura PDF, sia gli Image XObject. Nessun errore in console. Rasterizzare
# le figure non serve, perche' PNG e JPEG sono anch'essi Image XObject.
#
# LA SOLUZIONE. Le figure non sono piu' incluse come PDF ma disegnate con comandi
# PGF nativi (img-pgf/*.pgf, generati dal backend PGF di matplotlib e inseriti con
# \pgffig). Finiscono cosi' direttamente nel flusso di pagina, usano i font del
# documento e restano ricercabili. In piu' il PDF e' passato da 1.2 MB a ~410 KB.
#
# DUE DETTAGLI.
#   - Le matrici articolo usavano ax.imshow, che e' un raster: in irgood e' stato
#     sostituito con ax.pcolormesh, visivamente identico ma vettoriale.
#   - training_loss_mistral resta un \includegraphics: il suo .pgf pesa 2.9 MB e
#     satura la memoria di TeX. E' l'unica figura che nel viewer non si vedra',
#     insieme alle 16 strisce di colorbar, che matplotlib emette come raster.
#
# Per rigenerare le figure: vedi 2-analysis/, salvando in .pgf invece che in .pdf.
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
tectonic -X compile supplementary.tex --reruns 3
forms=$(python3 -c "import re;print(len(re.findall(rb'/Subtype\s*/Form',open('supplementary.pdf','rb').read())))")
words=$(pdftotext supplementary.pdf - | wc -w)
echo "supplementary.pdf: $(pdfinfo supplementary.pdf | grep -oP 'Pages:\s+\K\d+') pagine, $(du -h supplementary.pdf | cut -f1), $forms Form XObject, $words parole ricercabili"
