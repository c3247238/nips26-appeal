#!/usr/bin/env python3
"""Improved markdown to LaTeX converter - v3."""

import re
import sys
from pathlib import Path

MATH_PLACEHOLDER = '\x00MATH{}\x00'

def protect_math(text):
    math_blocks = []
    def replace(m):
        math_blocks.append(m.group(0))
        return MATH_PLACEHOLDER.format(len(math_blocks)-1)
    text = re.sub(r'\$\$[^$]*\$\$', replace, text)
    text = re.sub(r'\$[^$\n]+?\$', replace, text)
    return text, math_blocks

def restore_math(text, math_blocks):
    for i, math in enumerate(math_blocks):
        text = text.replace(MATH_PLACEHOLDER.format(i), math)
    return text

def escape_latex(text):
    chars = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
    }
    for ch, repl in chars.items():
        text = text.replace(ch, repl)
    return text

def process_inline(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
    return text

def convert_md_to_latex(md_text):
    lines = md_text.split('\n')
    output = []
    in_table = False
    table_lines = []
    in_list = False
    list_type = None
    title_set = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped and in_list:
            output.append(f'\\end{{{list_type}}}')
            output.append('')
            in_list = False
            list_type = None
            i += 1
            continue

        # Skip horizontal rules
        if stripped == '---' or stripped == '***' or re.match(r'^-{3,}$', stripped):
            i += 1
            continue

        # Document title: first # line that is NOT ##
        if stripped.startswith('# ') and not stripped.startswith('## ') and not title_set:
            title = stripped[2:].strip()
            t, math_blocks = protect_math(title)
            t = escape_latex(t)
            t = restore_math(t, math_blocks)
            output.append(f'\\title{{{t}}}')
            output.append('\\maketitle')
            title_set = True
            i += 1
            continue

        # Section (# )
        if stripped.startswith('# ') and not stripped.startswith('## '):
            title = stripped[2:].strip()
            t, math_blocks = protect_math(title)
            t = escape_latex(t)
            t = process_inline(t)
            t = restore_math(t, math_blocks)
            output.append(f'\\section{{{t}}}')
            i += 1
            continue

        # Subsection (## )
        if stripped.startswith('## ') and not stripped.startswith('### '):
            title = stripped[3:].strip()
            t, math_blocks = protect_math(title)
            t = escape_latex(t)
            t = process_inline(t)
            t = restore_math(t, math_blocks)
            output.append(f'\\subsection{{{t}}}')
            i += 1
            continue

        # Subsubsection (### )
        if stripped.startswith('### ') and not stripped.startswith('#### '):
            title = stripped[4:].strip()
            t, math_blocks = protect_math(title)
            t = escape_latex(t)
            t = process_inline(t)
            t = restore_math(t, math_blocks)
            output.append(f'\\subsubsection{{{t}}}')
            i += 1
            continue

        # Tables
        if '|' in stripped:
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            output.extend(convert_table(table_lines))
            in_table = False
            table_lines = []
            continue

        # Lists
        if re.match(r'^\s*[-*]\s', stripped):
            if not in_list:
                in_list = True
                list_type = 'itemize'
                output.append('\\begin{itemize}')
            item_text = re.sub(r'^\s*[-*]\s', '', stripped)
            t, math_blocks = protect_math(item_text)
            t = escape_latex(t)
            t = process_inline(t)
            t = restore_math(t, math_blocks)
            output.append(f'  \\item {t}')
            i += 1
            continue
        elif re.match(r'^\s*\d+\.\s', stripped):
            if not in_list:
                in_list = True
                list_type = 'enumerate'
                output.append('\\begin{enumerate}')
            item_text = re.sub(r'^\s*\d+\.\s', '', stripped)
            t, math_blocks = protect_math(item_text)
            t = escape_latex(t)
            t = process_inline(t)
            t = restore_math(t, math_blocks)
            output.append(f'  \\item {t}')
            i += 1
            continue

        # Regular paragraph
        if stripped:
            t, math_blocks = protect_math(stripped)
            t = escape_latex(t)
            t = process_inline(t)
            t = restore_math(t, math_blocks)
            output.append(t)
        else:
            output.append('')

        i += 1

    if in_table:
        output.extend(convert_table(table_lines))
    if in_list:
        output.append(f'\\end{{{list_type}}}')

    return '\n'.join(output)


def convert_table(lines):
    if len(lines) < 2:
        return []

    data_lines = [l for l in lines if not re.match(r'^\s*\|?\s*[-:]+\s*\|', l)]
    if not data_lines:
        return []

    header = data_lines[0]
    cells = [c.strip() for c in header.split('|') if c.strip()]
    num_cols = len(cells)
    col_spec = 'l' * num_cols

    output = ['\\begin{table}[ht]', '\\centering', '\\small']
    output.append(f'\\begin{{tabular}}{{{col_spec}}}')
    output.append('\\toprule')

    for i, line in enumerate(data_lines):
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if not cells:
            continue
        row_cells = []
        for c in cells:
            t, math_blocks = protect_math(c)
            t = escape_latex(t)
            t = process_inline(t)
            t = restore_math(t, math_blocks)
            row_cells.append(t)
        row = ' & '.join(row_cells) + ' \\\\\\'
        output.append(row)
        if i == 0 and len(data_lines) > 1:
            output.append('\\midrule')

    output.append('\\bottomrule')
    output.append('\\end{tabular}')
    output.append('\\caption{Experimental results}')
    output.append('\\end{table}')
    return output


def main():
    md_path = sys.argv[1] if len(sys.argv) > 1 else 'paper.md'
    text = Path(md_path).read_text(encoding='utf-8')

    latex_body = convert_md_to_latex(text)

    template = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}

"""

    template += latex_body
    template += "\n\n\\end{document}\n"

    out_path = Path(md_path).with_suffix('.tex')
    out_path.write_text(template, encoding='utf-8')
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
