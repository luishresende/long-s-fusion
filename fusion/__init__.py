"""Asymmetric fusion and normalization rules; nothing here knows about datasets.

    providers/   one venv per OCR engine, JSON-per-line protocol
    fusion/      the rules that combine the output of two engines  <- here
    experiments/ the batch runs that put that over corpora and measure

The modules:

    canon        canonical form for comparing: unifies encoding (same printed
                 glyph, different codepoints), never spelling
    common       the asymmetric fusion (matching by Levenshtein over the
                 strings), the `Rules` dataclass with the production flags, and
                 the metrics
"""
