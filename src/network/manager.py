from typing import List, Protocol
from .models import TrackingItem

class ProcessingCoordinator(Protocol):
    """
    Interface for the processing coordinator that the NetworkService interacts with.
    """
    
    def update_tracking(self, items: List[TrackingItem]):
        """Callback for when tracking information is updated."""
        ...

    def person_left(self, person_id: int) -> bool:
        """Callback for when a person leaves."""
        ...

    def start_simulation(self):
        """Starts the simulation (or real processing) background task."""
        ...

    def stop_simulation(self):
        """Stops the simulation (or real processing) background task."""
        ...

