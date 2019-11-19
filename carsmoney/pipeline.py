from abc import ABC, abstractmethod

class Pipeline(ABC):
    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Should load data for pipeline."""

    @abstractmethod
    def train(self, *args, **kwargs):
        """Should train on loaded data."""

    @abstractmethod
    def test(self, *args, **kwargs):
        """Should run inference on provided data."""

class Pipelines(dict):
    """Accessor for registered pipelines"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# This is a little non-conventional, but potentialy gives ways of experimenting
# quickly
pipelines = Pipelines()


def register(cls):
    pipelines[cls.__name__.lower()] = cls
