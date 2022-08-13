import numpy as np

from prometheus_crosswalk.src.stateful_abc import CrosswalkAlgorithm


class ParticleFilter(CrosswalkAlgorithm):
    """Algorithm based on the particle filter approach

    Check the link below for implementation:
    https://ros-developer.com/2019/04/10/parcticle-filter-explained-with-python-code-from-scratch/
    """

    def __init__(self):
        pass

    def sureness(self, frame: np.ndarray) -> float:
        return 0.0
