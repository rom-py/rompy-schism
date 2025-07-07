"""
SCHISM Module for ROMPY

This module provides interfaces and utilities for the ROMPY framework.
"""

from rompy.core.logging import get_logger

logger = get_logger(__name__)

# Import SCHISM components
from .config import SCHISMConfig
from .data import SCHISMData, SCHISMDataSflux, SCHISMDataWave
from .grid import SCHISMGrid

# Log module initialization
logger.debug("SCHISM module initialized")
