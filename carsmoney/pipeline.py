class Pipeline(object):
    def __init__(self):
        self.__class__


class Pipelines(dict):
    """Accessor for registered pipelines"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# This is a little non-conventional, but potentialy gives ways of experimenting
# quickly
pipeline = Pipelines()


def register(cls):
    pipelines[cls.__name__.lower()] = cls


# Do pipeline imports here
import segmentation
