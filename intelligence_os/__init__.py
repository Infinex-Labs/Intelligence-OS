"""Perceptual Memory System (POC) — a memory system that happens to have eyes.

Built in gated layers (see docs/intelligence_os/POC.md §4). Import order mirrors the
cascade: store (memory) is the product; everything else is glue around it.
"""
from .config import CONFIG, Config  # noqa: F401
from .store import Store  # noqa: F401

__all__ = ["CONFIG", "Config", "Store"]
