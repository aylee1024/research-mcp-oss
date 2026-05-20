# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Extract plain-text content from a TeX source directory (or single .tex file).

Strategy:
1. Find the main .tex file (prefer file containing \\begin{document}; fall back to largest .tex).
2. Inline any \\input{} and \\include{} directives.
3. Run pandoc with --to=plain --wrap=none to produce plain text. Math is rendered
   in plain Unicode form (e.g. $E = mc^2$ becomes ``E = mc²''), which is good for
   full-text search and embeddings but does not preserve LaTeX syntax.
4. Fall back to a regex-based TeX stripper if pandoc fails or produces too little
   output. The fallback stashes math environments and inline math before
   command-stripping and restores them intact at the end, so the fallback output
   preserves LaTeX math syntax (e.g. \\frac{a}{b}) whereas the pandoc output does not.

Output goes to stdout (parallel to process_pdf.py).

Usage:
    uv run process_tex.py <path_to_tex_file_or_directory>
"""

import re
import subprocess
import sys
from pathlib import Path


def find_main_tex(root: Path) -> Path | None:
    """Locate the main .tex file in a directory (or return the file itself)."""
    if root.is_file() and root.suffix == '.tex':
        return root
    if not root.is_dir():
        return None

    tex_files = sorted(root.rglob('*.tex'))
    if not tex_files:
        return None

    # Prefer a file containing \begin{document}
    candidates: list[tuple[int, Path]] = []
    for tex in tex_files:
        try:
            head = tex.read_text(errors='ignore')[:10000]
        except Exception:
            continue
        if r'\begin{document}' in head:
            candidates.append((tex.stat().st_size, tex))

    if candidates:
        return max(candidates, key=lambda x: x[0])[1]

    # Fall back to largest .tex by size
    return max(tex_files, key=lambda p: p.stat().st_size)


def inline_inputs(main_tex: Path, seen: set[Path] | None = None) -> str:
    """Recursively inline \\input{} and \\include{} directives."""
    if seen is None:
        seen = set()

    resolved = main_tex.resolve()
    if resolved in seen:
        return ''
    seen.add(resolved)

    try:
        text = main_tex.read_text(errors='ignore')
    except Exception:
        return ''

    base_dir = main_tex.parent

    def replacer(match: re.Match) -> str:
        cmd, fname = match.group(1), match.group(2)
        # Strip optional .tex extension
        fname = fname.strip()
        if not fname.endswith('.tex'):
            fname = fname + '.tex'
        child = base_dir / fname
        if not child.exists():
            # Try without the added .tex
            child = base_dir / match.group(2).strip()
        if child.exists() and child.is_file():
            return '\n' + inline_inputs(child, seen) + '\n'
        return ''  # dropped if not found

    pattern = re.compile(r'\\(input|include)\s*\{([^}]+)\}')
    return pattern.sub(replacer, text)


def pandoc_convert(tex_content: str) -> str | None:
    """Convert TeX to plain text via pandoc. Math is rendered in plain Unicode
    form rather than preserved as LaTeX. Returns None on failure.
    """
    try:
        result = subprocess.run(
            ['pandoc', '--from=latex', '--to=plain', '--wrap=none'],
            input=tex_content,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def regex_strip(tex: str) -> str:
    """Fallback: strip TeX commands with regex. Math environments and inline math
    ($...$, \\(...\\), \\[...\\], equation/align/eqnarray/gather/multline) are
    stashed as placeholders before command-stripping and restored at the end, so
    they survive with their TeX commands intact (e.g. \\frac{a}{b} stays as
    \\frac{a}{b}, not ``a b'').
    """
    # Remove comments
    tex = re.sub(r'(?<!\\)%[^\n]*', '', tex)
    # Drop preamble up to \begin{document}
    m = re.search(r'\\begin\{document\}', tex)
    if m:
        tex = tex[m.end():]
    # Drop everything after \end{document}
    m = re.search(r'\\end\{document\}', tex)
    if m:
        tex = tex[:m.start()]

    # Drop layout environments entirely (not math)
    for env in ('figure', 'table', 'tikzpicture', 'tabular'):
        tex = re.sub(
            rf'\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}',
            '',
            tex,
            flags=re.DOTALL,
        )

    # Stash math blocks before the command-stripping pass. Each stashed block
    # is replaced with a placeholder @@MATHN@@ that won't be touched by later
    # regex. At the end we re-substitute placeholders with original math.
    math_blocks: list[str] = []

    def _stash(replacement: str) -> str:
        idx = len(math_blocks)
        math_blocks.append(replacement)
        return f'@@MATH{idx}@@'

    # Display math environments → $$...$$
    for env in ('equation', 'align', 'eqnarray', 'gather', 'multline'):
        def _env_stash(match: re.Match, _env=env) -> str:
            content = match.group(1).strip()
            return _stash(f'\n$$\n{content}\n$$\n')
        tex = re.sub(
            rf'\\begin\{{{env}\*?\}}(.*?)\\end\{{{env}\*?\}}',
            _env_stash,
            tex,
            flags=re.DOTALL,
        )
    # \[ ... \] display math
    tex = re.sub(
        r'\\\[(.*?)\\\]',
        lambda m: _stash(f'\n$$\n{m.group(1).strip()}\n$$\n'),
        tex,
        flags=re.DOTALL,
    )
    # \( ... \) inline math
    tex = re.sub(
        r'\\\((.*?)\\\)',
        lambda m: _stash(f'${m.group(1)}$'),
        tex,
        flags=re.DOTALL,
    )
    # $...$ inline math (but not $$...$$ display math; match single $ pairs only)
    tex = re.sub(
        r'(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)',
        lambda m: _stash(f'${m.group(1)}$'),
        tex,
    )

    # Section headers
    tex = re.sub(r'\\(?:sub)*section\*?\s*\{([^}]*)\}', r'\n\n## \1\n\n', tex)
    tex = re.sub(r'\\chapter\*?\s*\{([^}]*)\}', r'\n\n# \1\n\n', tex)
    # Environments we want to keep contents of
    for env in ('abstract', 'theorem', 'lemma', 'definition', 'proposition', 'corollary', 'proof', 'remark'):
        tex = re.sub(rf'\\begin\{{{env}\*?\}}', f'\n\n[{env.upper()}] ', tex)
        tex = re.sub(rf'\\end\{{{env}\*?\}}', '\n', tex)
    # Citations and refs
    tex = re.sub(r'\\cite[a-z]*\*?\s*(?:\[[^\]]*\])?\s*\{[^}]+\}', '[CITE]', tex)
    tex = re.sub(r'\\ref\*?\s*\{[^}]+\}', '[REF]', tex)
    tex = re.sub(r'\\label\s*\{[^}]+\}', '', tex)
    # Basic text formatting
    tex = re.sub(r'\\text(?:bf|it|sf|sl|rm|tt)\s*\{([^}]*)\}', r'\1', tex)
    tex = re.sub(r'\\emph\s*\{([^}]*)\}', r'\1', tex)
    # Remove unknown backslash commands (math was stashed, so this only
    # touches non-math LaTeX)
    tex = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?', ' ', tex)
    # Clean up braces (math is stashed, so this doesn't touch math braces)
    tex = re.sub(r'[{}]', '', tex)
    # Collapse whitespace
    tex = re.sub(r'\s+\n', '\n', tex)
    tex = re.sub(r'\n{3,}', '\n\n', tex)
    tex = re.sub(r'[ \t]+', ' ', tex)

    # Restore stashed math blocks
    for i, block in enumerate(math_blocks):
        tex = tex.replace(f'@@MATH{i}@@', block)

    return tex.strip()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: process_tex.py <path_to_tex_file_or_directory>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: path not found: {target}", file=sys.stderr)
        sys.exit(1)

    main_tex = find_main_tex(target)
    if main_tex is None:
        print(f"Error: no .tex file found in {target}", file=sys.stderr)
        sys.exit(1)

    tex_content = inline_inputs(main_tex)
    if not tex_content.strip():
        print(f"Error: empty TeX content from {main_tex}", file=sys.stderr)
        sys.exit(1)

    output = pandoc_convert(tex_content)
    if output is None or len(output.strip()) < 100:
        # Fallback to regex stripper
        output = regex_strip(tex_content)

    sys.stdout.write(output)


if __name__ == '__main__':
    main()
