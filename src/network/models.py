"""
Pydantic models for the Network API, based on webservice-api-spec.md.
"""
from typing import List
from pydantic import BaseModel, Field

class TrackingItem(BaseModel):
    """
    Represents a single person's tracking information.
    """
    id: int = Field(..., description="A unique integer identifier for the person.")
    angle: float = Field(..., description="The detected angle in degrees.")

class LeaveNotification(BaseModel):
    """
    Represents a notification that a person has left.
    """
    id: int = Field(..., description="The unique identifier of the person who left.")

class AngleEnergy(BaseModel):
    """
    Represents the angle and energy of a detected sound source.
    """
    angle: float = Field(..., description="The detected angle in degrees.")
    energy: float = Field(..., description="The energy level of the detected sound (float32).")

class DeviceIDItem(BaseModel):
    """
    Represents a client device ID update request.
    """
    device_id: str = Field(..., description="The unique identifier for the client device.")

class TrackingUpdateResponse(BaseModel):
    """
    Response model for tracking updates, including session information.
    """
    id: int = Field(..., description="The person ID.")
    session_uuid: str = Field(..., description="The unique UUID for the tracking session.")
    log_directory_name: str = Field(..., description="The name of the log directory for this session.")
