"""Evaluation-realignment analysis track (note ``LiquidityML_Portfolio_Issue``).

This package implements the note's section-4 evaluation variations on top of the
existing formal experiment predictions. It is deliberately separate from
``src.analysis.formal`` so the realigned evaluation never collides with the
formalanalysis pipeline; predictions are read from the formalanalysis experiment
cache and outputs are written under ``outputs/eval_realignment/``.
"""
