"""CoFrame: self-validating adaptive frame meshes for video DiTs."""

from .controller import CoFrameController, ControllerConfig, Selection
from .mesh import interpolate_frame_values, uniform_indices
from .selectors import rhyme_budgeted_indices, rhyme_sequential_threshold

__all__ = [
    "CoFrameController",
    "ControllerConfig",
    "Selection",
    "interpolate_frame_values",
    "uniform_indices",
    "rhyme_budgeted_indices",
    "rhyme_sequential_threshold",
]

__version__ = "0.1.0"
