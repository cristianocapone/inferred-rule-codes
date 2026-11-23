"""Main-text word count: abstract through the end of the Discussion, without figures, the box or Methods.

Journals count the narrative, not the captions, the reader box or the Methods, so this strips those first
and then runs detex. Usage: python analysis/wordcount.py [main_nature.tex]
"""
import os, re, subprocess, sys

from paths import PAPER

p = sys.argv[1] if len(sys.argv) > 1 else str(PAPER / "main_nature.tex")
s = open(p).read()
body = s[s.index(r"\begin{abstract}"):s.index(r"\begin{methods}")]
for env in ("figure", "floatingbox", "table"):
    body = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), "", body, flags=re.S)
out = subprocess.run(["detex"], input=body, capture_output=True, text=True).stdout
print(len(out.split()), "words in the main text")
