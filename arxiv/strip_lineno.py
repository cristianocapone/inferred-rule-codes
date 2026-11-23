"""Remove the review line numbers from a copy of the Nature Communications layout, for the arXiv upload.

The manuscript carries `lineno` so that reviewers can point at a line; a preprint should not show them.
Usage: python strip_lineno.py <main.tex>
"""
import sys

p = sys.argv[1]
s = open(p).read()
s = s.replace("\\usepackage{graphicx,amsmath,xcolor,float,setspace,lineno,titlesec,caption}",
              "\\usepackage{graphicx,amsmath,xcolor,float,setspace,titlesec,caption}")
s = s.replace("\\linenumbers\n", "")
assert "lineno" not in s and "\\linenumbers" not in s, "line numbers still present"
open(p, "w").write(s)
print("line numbers stripped from", p)
