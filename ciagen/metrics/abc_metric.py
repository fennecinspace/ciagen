from abc import ABC, abstractmethod


class QualityMetric(ABC):
    """Abstract base class for quality metrics."""

    @classmethod
    @abstractmethod
    def allows_for_gpu(cls) -> bool:
        """Whether this metric supports GPU computation."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the name of this metric."""
        ...
