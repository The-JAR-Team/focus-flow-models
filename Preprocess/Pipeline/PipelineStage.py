# PipelineStage.py
from abc import ABC, abstractmethod


class PipelineStage(ABC):
    @abstractmethod
    def process(self, data):
        """Process input data and return the transformed output."""
        pass
