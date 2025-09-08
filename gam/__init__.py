"""
GAM: Gradient norm Aware Minimization

This package implements the GAM optimizer for improved generalization in neural networks.
"""

from .gam import GAM
from .smooth_cross_entropy import smooth_crossentropy

__all__ = ['GAM', 'smooth_crossentropy']