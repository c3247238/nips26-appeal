#!/usr/bin/env python3
"""Simple markdown to LaTeX converter for NeurIPS papers."""

import re
import sys
from pathlib import Path

def md2latex(text):
    lines = text.split('\n')
    output = []
    in_table = False
    table_lines = []
    in_code = False
    code_lines = []
    in_list = False
    list_type = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Code blocks
        if stripped.startswith('```'):
            if in_code:
                in_code = False
                output.append('\\end{verbatim}')
            else:
                in_code = True
                output.append('\\begin{verbatim}')
            i += 1
            continue

        if in_code:
            output.append(line)
            i += 1
            continue

        # Tables
        if '|' in stripped and not in_table:
            in_table = True
            table_lines = [stripped]
            i += 1
            continue
        elif in_table:
            if '|' in stripped:
                table_lines.append(stripped)
                i += 1
                continue
            else:
                # End table
                output.extend(convert_table(table_lines))
                in_table = False
                table_lines = []
                continue

        # Headers
        if stripped.startswith('# '):
            title = stripped[2:].strip()
            output.append(f'\\title{{{escape_latex(title)}}}')
        elif stripped.startswith('## '):
            title = stripped[3:].strip()
            output.append(f'\\section{{{escape_latex(title)}}}')
        elif stripped.startswith('### '):
            title = stripped[4:].strip()
            output.append(f'\\subsection{{{escape_latex(title)}}}')
        elif stripped.startswith('#### '):
            title = stripped[5:].strip()
            output.append(f'\\subsubsection{{{escape_latex(title)}}}')
        # Lists
        elif re.match(r'^\s*[-*]\s', stripped):
            if not in_list:
                in_list = True
                list_type = 'itemize'
                output.append('\\begin{itemize}')
            item_text = re.sub(r'^\s*[-*]\s', '', stripped)
            output.append(f'  \\item {inline_format(item_text)}')
        elif re.match(r'^\s*\d+\.\s', stripped):
            if not in_list:
                in_list = True
                list_type = 'enumerate'
                output.append('\\begin{enumerate}')
            item_text = re.sub(r'^\s*\d+\.\s', '', stripped)
            output.append(f'  \\item {inline_format(item_text)}')
        else:
            if in_list and not stripped:
                in_list = False
                output.append(f'\\end{{{list_type}}}')
                list_type = None

            if stripped:
                output.append(inline_format(stripped))
            elif output and output[-1] != '':
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

    # Remove separator line (---|---)
    data_lines = [l for l in lines if not re.match(r'^\s*\|?\s*[-:]+\s*\|', l)]
    if not data_lines:
        return []

    output = ['\\begin{table}[h]', '\\centering', '\\begin{tabular}{' + 'l' * 10 + '}']
    output.append('\\toprule')

    for i, line in enumerate(data_lines):
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if not cells:
            continue
        row = ' & '.join(escape_latex(c) for c in cells) + ' \\\\\\'
        output.append(row)
        if i == 0:
            output.append('\\midrule')

    output.append('\\bottomrule')
    output.append('\\end{tabular}')
    output.append('\\end{table}')
    return output


def escape_latex(text):
    # Don't escape math content
    if '$' in text:
        parts = []
        in_math = False
        for part in text.split('$'):
            if in_math:
                parts.append(f'${part}$')
            else:
                parts.append(_escape_plain(part))
            in_math = not in_math
        return ''.join(parts)
    return _escape_plain(text)


def _escape_plain(text):
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('^', '\\textasciicircum{}')
    return text


def inline_format(text):
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    # Italic
    text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
    # Citations [Author, Year] -> \citep{}
    text = re.sub(r'\[([^\]]+)\]', lambda m: f'\\citep{{{m.group(1).replace(" ", "").replace(",", "")}}}', text)
    return escape_latex(text)


def main():
    md_path = sys.argv[1] if len(sys.argv) > 1 else 'paper.md'
    text = Path(md_path).read_text(encoding='utf-8')

    latex_body = md2latex(text)

    template = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}

"""

    template += latex_body
    template += "\n\n\\bibliographystyle{plainnat}\n\\end{document}\n"

    out_path = Path(md_path).with_suffix('.tex')
    out_path.write_text(template, encoding='utf-8')
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
