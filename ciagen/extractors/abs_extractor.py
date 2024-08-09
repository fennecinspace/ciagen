from abc import ABC


class ExtractorABC(ABC):
    def __str__(self):
        return self.name
