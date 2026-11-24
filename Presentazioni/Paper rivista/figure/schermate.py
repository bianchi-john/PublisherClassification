"""Le «schermate» delle tabelle e delle figure del paper per le slide: ogni tabella o figura di main.tex viene composta
da sola (stesso preambolo, stesso numero e stessa didascalia del PDF), poi trasformata in PNG e ritagliata.
I numeri delle tabelle, delle sezioni e delle citazioni vengono da main.aux: prima si compila il paper
(latexmk -pdf main.tex in Paper/Journal), poi:  python figure/schermate.py
Uscita: figure/schermate/<etichetta>.png (es. tab_main.png, fig_curves.png)."""
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

QUI = Path(__file__).resolve().parent
PAPER = QUI.parents[2] / "Paper" / "Journal"
OUT = QUI / "schermate"
OUT.mkdir(exist_ok=True)

tex = (PAPER / "main.tex").read_text()
aux = (PAPER / "main.aux").read_text()
numero = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))
cita = dict(re.findall(r"\\bibcite\{([^}]+)\}\{\{(\d+)\}", aux))
preambolo = tex[:tex.index(r"\begin{document}")]


def senza_rimandi(s):
    """\\ref e \\cite diventano i numeri del paper (le schermate si compongono senza il resto del documento)."""
    s = re.sub(r"\\(?:ref|eqref)\{([^}]+)\}", lambda m: numero.get(m.group(1), "?"), s)
    return re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(cita.get(k.strip(), "?") for k in m.group(1).split(",")) + "]", s)


def ritaglia(png, margine=12):
    im = Image.open(png).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    im.crop((max(box[0] - margine, 0), max(box[1] - margine, 0), min(box[2] + margine, im.width),
             min(box[3] + margine, im.height))).save(png)


for m in re.finditer(r"\\begin\{(table|figure)\}(\[[^\]]*\])?(.*?)\\end\{\1\}", tex, flags=re.S):
    tipo, corpo = m.group(1), m.group(3)
    etichetta = re.search(r"\\label\{([^}]+)\}", corpo).group(1)
    n = int(numero[etichetta])
    doc = (preambolo + "\\begin{document}\\pagestyle{empty}\n"
           + f"\\setcounter{{{tipo}}}{{{n - 1}}}\n\\begin{{{tipo}}}[h!]{senza_rimandi(corpo)}\\end{{{tipo}}}\n\\end{{document}}\n")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "f.tex").write_text(doc)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"-output-directory={tmp}", str(Path(tmp) / "f.tex")],
                           cwd=PAPER, capture_output=True, text=True)
        pdf = Path(tmp) / "f.pdf"
        assert pdf.exists(), f"{etichetta}: {r.stdout[-800:]}"
        nome = OUT / etichetta.replace(":", "_")
        subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", "-f", "1", "-l", "1", str(pdf), str(nome)], check=True)
    ritaglia(nome.with_suffix(".png"))
    print(f"{tipo} {n}: {nome.name}.png")
