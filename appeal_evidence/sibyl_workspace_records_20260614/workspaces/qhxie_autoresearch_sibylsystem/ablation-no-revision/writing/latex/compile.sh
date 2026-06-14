#!/bin/bash
# Compile the LaTeX paper to PDF
# Usage: bash compile.sh

cd "$(dirname "$0")"

echo "Compiling main.tex to PDF..."
latexmk -pdf main.tex

echo "Done. Output: main.pdf"