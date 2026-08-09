"""CoFrame: self-validating adaptive frame meshes for video diffusion."""

from .config import CoFrameConfig
from .controller import AdaptiveMeshController
from .selection import rhyme_select, uniform_select

__all__ = ["AdaptiveMeshController", "CoFrameConfig", "rhyme_select", "uniform_select"]
__version__ = "0.1.0"
