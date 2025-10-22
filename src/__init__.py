"""
Resume RAG System - Main package
"""

from .pipeline import ResumeRAGPipeline
from . import config

__version__ = "0.1.0"

__all__ = ["ResumeRAGPipeline", "config"]