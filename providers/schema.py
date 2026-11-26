"""Output contract of the providers.

Every engine returns a `Result`, whatever role it is playing. A field an engine
does not produce stays `None`/empty rather than disappearing, so the consumer
never needs to know which engine it came from.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Char:
    """One character: `x0`/`x1` on the line image, `t0`/`t1` frame indices with
    exclusive end, `alternatives` the other candidates without the winner."""
    text: str
    prob: Optional[float] = None
    x0: Optional[float] = None
    x1: Optional[float] = None
    t0: Optional[int] = None
    t1: Optional[int] = None
    alternatives: list = field(default_factory=list)


@dataclass
class Line:
    """One recognized line; `variants` are the Calamari fold readings."""
    text: str
    confidence: Optional[float] = None
    chars: list = field(default_factory=list)
    image_path: Optional[str] = None
    box: Optional[list] = None
    variants: list = field(default_factory=list)


@dataclass
class Result:
    provider: str
    lines: list = field(default_factory=list)
    page_path: Optional[str] = None
    elapsed: float = 0.0
    warnings: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
