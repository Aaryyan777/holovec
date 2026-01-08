# Expose key components to the top level
from .core import HyperVector
from .encoders import TextEncoder, ProjectionEncoder
from .memory import AssociativeMemory
from .learning import PerceptronClassifier

# Define what happens on 'from holovec import *'
__all__ = [
    "HyperVector",
    "TextEncoder",
    "ProjectionEncoder",
    "AssociativeMemory",
    "PerceptronClassifier",
]

# Version
__version__ = "0.1.0"