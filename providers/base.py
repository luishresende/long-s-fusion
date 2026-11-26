"""Contract shared by the OCR engines, each in its own venv behind a subprocess."""
from abc import ABC, abstractmethod


class Provider(ABC):
    name: str = "provider"

    @abstractmethod
    def load(self) -> None:
        """Load the model. Called once, before the first `predict`."""

    @abstractmethod
    def predict(self, request: dict) -> dict:
        """Process a deserialized request; input and output are JSON-serializable."""
