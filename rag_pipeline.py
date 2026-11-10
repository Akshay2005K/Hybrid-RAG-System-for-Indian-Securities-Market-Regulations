"""Backward-compatible import shim for the src package."""

from src.rag_stock_market.pipeline import *
from src.rag_stock_market.pipeline import __all__ as _pipeline_all

__all__ = _pipeline_all
