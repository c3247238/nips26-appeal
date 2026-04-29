TEX = sibyl_harness_position

.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode $(TEX).tex

clean:
	latexmk -C $(TEX).tex
