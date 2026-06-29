"""Compatibility module for the requested Haptos result types.

Runtime code imports from haptos_types.py because Python already has a standard
library module named types. Keeping this file preserves the requested project
structure without making imports ambiguous.
"""

from haptos_types import BBox, Detection, FrameResult

__all__ = ["BBox", "Detection", "FrameResult"]
