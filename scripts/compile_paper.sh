#!/bin/bash
# Clean full compile of the paper (single document since 2026-07-14: the
# former Internet Appendix is merged into the main appendices).
# Always deletes auxiliary files first: a compile interrupted mid-write
# leaves a truncated .aux/.lot that makes every later pass fail with
# "File ended while scanning" / "invalid character" errors.
#
#   bash scripts/compile_paper.sh       # Main.pdf
set -euo pipefail
cd "$(dirname "$0")/../paper"

build () {
    local doc="$1"
    rm -f "${doc}.aux" "${doc}.out" "${doc}.toc" "${doc}.lof" "${doc}.lot" \
          "${doc}.bbl" "${doc}.blg" "${doc}.log"
    pdflatex -interaction=nonstopmode "${doc}.tex" > /dev/null
    bibtex "${doc}" > /dev/null
    pdflatex -interaction=nonstopmode "${doc}.tex" > /dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "${doc}.tex" > /dev/null
    local pages warns over
    pages=$(pdfinfo "${doc}.pdf" | awk '/^Pages/{print $2}')
    warns=$(grep -c "undefined\|multiply" "${doc}.log" || true)
    over=$(grep -c "Overfull" "${doc}.log" || true)
    echo "${doc}.pdf: ${pages} pages, ${warns} reference warnings, ${over} overfull boxes"
}

build Main
