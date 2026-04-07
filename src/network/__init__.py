"""
This package contains the network service for the DOA audio processing system.
"""
from .service import NetworkService
from .manager import ProcessingCoordinator

__all__ = ["NetworkService", "ProcessingCoordinator"]
