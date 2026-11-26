"""La «schermata» del prompt dello zero-shot: il blocco quote dell'appendice F di main.tex, composto da solo con il
preambolo del paper (come figure/schermate.py fa con tabelle e figure), poi PNG ritagliato.
Uscita: figure/schermate/prompt.png. Uso:  python schermata_prompt.py"""
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

QUI = Path(__file__).resolve().parent
PAPER = QUI.parents[2] / "Paper" / "Journal"
tex = (PAPER / "main.tex").read_text()
preambolo = tex[:tex.index(r"\begin{document}")]
appendice = tex[tex.index(r"\label{app:prompt}"):]
quote = re.search(r"\\begin\{quote\}(.*?)\\end\{quote\}", appendice, flags=re.S).group(1)    # senza il rientro
doc = (preambolo + "\\begin{document}\\pagestyle{empty}\n\\noindent\\fbox{\\begin{minipage}{0.97\\linewidth}\n"
       + quote + "\n\\end{minipage}}\n\\end{document}\n")
uscita = QUI / "schermate" / "prompt"
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "f.tex").write_text(doc)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", f"-output-directory={tmp}", str(Path(tmp) / "f.tex")],
                   cwd=PAPER, capture_output=True, text=True)
    subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", "-f", "1", "-l", "1", str(Path(tmp) / "f.pdf"),
                    str(uscita)], check=True)
im = Image.open(uscita.with_suffix(".png")).convert("RGB")
box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
im.crop((box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8)).save(uscita.with_suffix(".png"))
print(uscita.with_suffix(".png"), Image.open(uscita.with_suffix(".png")).size)
