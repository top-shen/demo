"""Local VerbalTS training package.

Keeping this directory as an explicit package prevents an unrelated top-level
``train.py`` on ``PYTHONPATH`` (for example LLaMA-Factory's entry point) from
shadowing ``train.trainer``.
"""
