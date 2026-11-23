"""Build the Nature Communications version of the manuscript from main_nature.tex.

The manuscript is written for nature.cls (Nature's preprint conventions: affiliations environment, methods
environment, addendum, superscript citations). Nature Communications asks instead for a plain article with the
sections Introduction, Results, Discussion and Methods, the statements at the end, figures with their legends, and
line numbers for review. Rather than keep two copies of the text, this script rewrites the file: preamble, the few
environments nature.cls defines, and the section headings. Run it after editing main_nature.tex.

    python make_nc.py && pdflatex main_nc.tex && bibtex main_nc && pdflatex main_nc.tex && pdflatex main_nc.tex
"""

from paths import ROOT, DATA, PAPER, FIGURES, TABLES, RESULTS
import os, re

HERE = str(PAPER)
SRC, DST = os.path.join(HERE, "main_nature.tex"), os.path.join(HERE, "main_nc.tex")

PREAMBLE = r"""%% Journal layout (the Nature Communications submission), generated from main_nature.tex
%% by make_nc.py -- do not edit by hand. Typeset closer to how a Nature-family article reads: a Times-like
%% serif body, sans-serif section headings, bold run-in captions, single line spacing, and the figures set
%% in the text beside the passage that discusses them rather than collected after the references.
\documentclass[10pt]{article}
\usepackage[a4paper,margin=2.05cm,top=2.2cm,bottom=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{helvet}
\usepackage{graphicx,amsmath,xcolor,float,setspace,lineno,titlesec,caption}
\usepackage[super,comma,sort&compress]{natbib}
\input{readerbox}
\usepackage[hidelinks]{hyperref}
\usepackage{etoolbox,needspace}
%% a heading alone at the foot of a page reads as a mistake; keep a few lines of its section with it
\pretocmd{\section}{\Needspace{4\baselineskip}}{}{}
\pretocmd{\subsection}{\Needspace{3\baselineskip}}{}{}
\newcommand{\todo}[1]{\textcolor{red}{[#1]}}

%% The figures are close to a full page each. Left at their default limits LaTeX cannot fit one on a page
%% that also carries text, defers them all, and prints them after the references; forced into place with [H]
%% they push a page break and leave the rest of a page empty. These limits let a large figure take a page of
%% its own as soon as it is introduced, so it stays next to the text that discusses it and no page is left short.
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.7}
\bibliographystyle{naturemag}
\renewcommand{\thefigure}{\arabic{figure}}
\definecolor{ncteal}{HTML}{0B6E71}
\titleformat{\section}{\sffamily\bfseries\large\color{ncteal}}{}{0pt}{}[\vspace{1pt}\color{ncteal}\titlerule\vspace{2pt}]
\titlespacing{\section}{0pt}{11pt}{4pt}
\titleformat{\subsection}{\sffamily\bfseries\normalsize}{}{0pt}{}
\titlespacing{\subsection}{0pt}{8pt}{3pt}
\captionsetup{font=small,labelfont={sf,bf},labelsep=period,skip=5pt}
\setlength{\parskip}{1pt}
\setlength{\floatsep}{9pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{11pt plus 2pt minus 3pt}
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\linespread{1.0}
\linenumbers
\title{\sffamily\bfseries\Large Non-linguistic communication of abstract rules emerges from reward}
\author{\normalfont Cristiano Capone\\[2pt]\normalsize Quantitative Health Models and Bioinspired Technologies Lab, Istituto Superiore di Sanit\`a, Rome, Italy\\[1pt]\small\texttt{cristiano.capone@iss.it}}
\date{}
\begin{document}
\maketitle
\vspace{-8pt}{\color{ncteal}\hrule height 1pt}\vspace{4pt}
"""


def main():
    s = open(SRC).read()
    body = s[s.index(r"\begin{abstract}"):s.index(r"\end{document}")]
    # abstract, and the introduction Nature leaves unheaded
    body = body.replace(r"\begin{abstract}", "\\section*{Abstract}\n\\noindent ")
    body = body.replace(r"\end{abstract}", "\n\n\\section*{Introduction}")
    # methods: a section, its parts subsections, and the two statements as sections of their own
    body = body.replace(r"\begin{methods}", r"\section*{Methods}").replace(r"\end{methods}", "")
    body = re.sub(r"\\subsection\{", r"\\subsection*{", body)
    body = body.replace(r"\subsection*{Data availability}", r"\section*{Data availability}")
    body = body.replace(r"\subsection*{Code availability}", r"\section*{Code availability}")
    # end matter: Nature's addendum becomes the statements Nature Communications asks for
    m = re.search(r"\\begin\{addendum\}(.*?)\\end\{addendum\}", body, re.S)
    if m:
        items = re.findall(r"\\item\[(.*?)\]\s*(.*?)(?=\\item\[|\Z)", m.group(1), re.S)
        end = "".join(f"\n\\section*{{{t.strip()}}}\n{v.strip()}\n" for t, v in items)
        body = body[:m.start()] + end + body[m.end():]
    # the bibliography sits after the statements in this format
    body = body.replace("\\bibliography{refs}\n", "")
    open(DST, "w").write(PREAMBLE + body + "\n\\bibliography{refs}\n\n\\end{document}\n")
    print(f"wrote {os.path.basename(DST)}")


if __name__ == "__main__":
    main()
