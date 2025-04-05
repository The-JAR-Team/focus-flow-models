from abc import ABC, abstractmethod


class PipelineStage(ABC):
    @abstractmethod
    def process(self, data, verbose=True):
        """
        Process input data and return the transformed output.

        Parameters:
          data: The input data.
          verbose (bool): If True, the stage prints detailed status messages.

        Returns:
          The processed output.
        """
        pass
