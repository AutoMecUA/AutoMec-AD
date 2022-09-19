"""Contains the abstract base class (abc) for defining stateful algorithms

"""

from abc import ABC, abstractmethod

import numpy as np


class CrosswalkAlgorithm(ABC):

    @abstractmethod
    def sureness(self, frame: np.ndarray):
        pass
