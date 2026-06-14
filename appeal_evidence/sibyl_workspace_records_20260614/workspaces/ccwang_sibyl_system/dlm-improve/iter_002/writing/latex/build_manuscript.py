from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path("/Users/cwan0785/sibyl-system")
WORKSPACE = Path("/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current")
WRITING_DIR = WORKSPACE / "writing"
LATEX_DIR = WRITING_DIR / "latex"
FIGURES_DIR = LATEX_DIR / "figures"
TEMPLATE_DIR = ROOT / "sibyl" / "templates" / "neurips_2024"

TITLE_LINE_RE = re.compile(r"^#\s+(.*)$")
HEADING_RE = re.compile(r"^(#{2,4})\s+(.*)$")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
INLINE_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_inline(text: str) -> str:
    placeholder_map: dict[str, str] = {}

    def stash(pattern: re.Pattern[str], wrapper: str, source: str) -> str:
        def repl(match: re.Match[str]) -> str:
            token = f"@@TOKEN{len(placeholder_map)}@@"
            placeholder_map[token] = wrapper.format(escape_latex(match.group(1)))
            return token

        return pattern.sub(repl, source)

    text = stash(INLINE_CODE_RE, r"\texttt{{{}}}", text)
    text = stash(INLINE_BOLD_RE, r"\textbf{{{}}}", text)
    text = escape_latex(text)
    for token, value in placeholder_map.items():
        text = text.replace(escape_latex(token), value)
    return text


def split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def parse_markdown(md_text: str) -> tuple[str, str, list[str]]:
    blocks = split_blocks(md_text)
    title = ""
    abstract_parts: list[str] = []
    body_blocks: list[str] = []
    in_abstract = False
    inserted_protocol_figure = False

    for block in blocks:
        first = block[0]
        title_match = TITLE_LINE_RE.match(first)
        if title_match:
            title = title_match.group(1).strip()
            continue

        heading_match = HEADING_RE.match(first)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if level == 2 and heading_text.lower() == "abstract":
                in_abstract = True
                continue
            in_abstract = False
            if level == 2:
                body_blocks.append(f"\\section{{{escape_latex(heading_text)}}}")
            elif level == 3:
                body_blocks.append(f"\\subsection{{{escape_latex(heading_text)}}}")
            else:
                body_blocks.append(f"\\subsubsection{{{escape_latex(heading_text)}}}")
            continue

        if all(line.startswith("- ") for line in block):
            items = "\n".join(
                f"\\item {format_inline(line[2:].strip())}" for line in block
            )
            rendered = "\\begin{itemize}\n" + items + "\n\\end{itemize}"
        else:
            paragraph = " ".join(line.strip() for line in block)
            rendered = format_inline(paragraph)

        if in_abstract:
            abstract_parts.append(rendered)
            continue

        body_blocks.append(rendered)
        if (
            not inserted_protocol_figure
            and body_blocks
            and body_blocks[-2:-1] == [r"\subsection{3.1 Observer, controller, and runtime are different objects}"]
            and (FIGURES_DIR / "signal_audit_protocol.pdf").exists()
        ):
            body_blocks.append(
                "Figure~\\ref{fig:signal-audit-protocol} summarizes the protocol objects "
                "that separate observer signals, realized controller outcomes, and runtime conditions."
            )
            body_blocks.append(
                "\n".join(
                    [
                        r"\begin{figure}[t]",
                        r"\centering",
                        r"\includegraphics[width=0.92\linewidth]{figures/signal_audit_protocol.pdf}",
                        r"\caption{Protocol view of the observer signal $d(s)$, realized controller outcome $g(s)$, and runtime metadata required for an honest compute audit.}",
                        r"\label{fig:signal-audit-protocol}",
                        r"\end{figure}",
                    ]
                )
            )
            inserted_protocol_figure = True

    abstract = "\n\n".join(abstract_parts).strip()
    return title, abstract, body_blocks


def render_protocol_figure(output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    boxes = [
        (0.4, 1.05, 3.2, 1.4, "#d7e8ba", "Observer\n$d(s)$\nerror-likelihood signal"),
        (5.1, 1.05, 3.4, 1.4, "#f8d8a8", "Controller\nrevision policy\nrealized outcome $g(s)$"),
        (10.0, 0.75, 3.0, 2.0, "#c9ddf2", "Runtime metadata\nbackend\ncompile path\nsafe batch size"),
    ]
    for x, y, w, h, color, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.2,
            edgecolor="#2f3b45",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    arrows = [
        ((3.75, 1.75), (4.95, 1.75)),
        ((8.6, 2.1), (9.85, 2.1)),
        ((11.5, 0.75), (7.0, 0.45)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.2,
                color="#2f3b45",
                connectionstyle="arc3,rad=0.0",
            )
        )

    ax.text(
        6.75,
        0.18,
        "Audit rule: observer quality, controller gain, and runtime fairness must be reported separately.",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_main_tex(title: str, abstract: str, body_blocks: list[str]) -> str:
    body = "\n\n".join(body_blocks)
    abstract_text = abstract or "To be completed."
    return f"""\\documentclass{{article}}

\\usepackage[final]{{neurips_2024}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{hyperref}}
\\usepackage{{url}}
\\usepackage{{booktabs}}
\\usepackage{{amsfonts}}
\\usepackage{{nicefrac}}
\\usepackage{{microtype}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{enumitem}}

\\title{{{escape_latex(title)}}}
\\author{{Anonymous Authors}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{abstract_text}
\\end{{abstract}}

{body}

\\bibliography{{references}}
\\bibliographystyle{{plainnat}}

\\end{{document}}
"""


def build_references_bib(output_path: Path) -> None:
    references_json = next(WORKSPACE.rglob("references.json"), None)
    if references_json is None:
        output_path.write_text("% No references.json was available in the workspace.\n", encoding="utf-8")
        return
    output_path.write_text("% Reference export is intentionally empty pending citation normalization.\n", encoding="utf-8")


def main() -> None:
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "neurips_2024.sty", LATEX_DIR / "neurips_2024.sty")
    render_protocol_figure(FIGURES_DIR / "signal_audit_protocol.pdf")

    paper_path = WRITING_DIR / "paper.md"
    title, abstract, body_blocks = parse_markdown(paper_path.read_text(encoding="utf-8"))
    (LATEX_DIR / "main.tex").write_text(build_main_tex(title, abstract, body_blocks), encoding="utf-8")
    build_references_bib(LATEX_DIR / "references.bib")


if __name__ == "__main__":
    main()
