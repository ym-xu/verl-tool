"""Base class for dataset processors."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import datasets


class DatasetProcessor(ABC):
    """Base class for all dataset processors."""

    def __init__(self, name: str, config: dict, output_dir: str):
        self.name = name
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    def load(self) -> datasets.Dataset:
        """Load raw dataset."""
        ...

    @abstractmethod
    def process(self) -> Dict[str, datasets.Dataset]:
        """
        Process dataset into unified format.
        Returns dict mapping task_type → Dataset.
        E.g., {"vqa": Dataset, "gnd": Dataset}
        """
        ...

    def get_stats(self) -> dict:
        """Return processing statistics."""
        return {"name": self.name, "config": self.config}
