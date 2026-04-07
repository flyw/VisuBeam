import os
from typing import Optional

DEVICE_ID_FILE = ".client_device_id"


def save_client_device_id(device_id: str):
    """
    Saves the client device ID to a hidden file in the project root.
    Overwrites the file if it exists.
    """
    with open(DEVICE_ID_FILE, "w", encoding="utf-8") as f:
        f.write(device_id.strip())


def load_client_device_id() -> Optional[str]:
    """
    Loads the client device ID from the hidden file.
    Returns None if the file does not exist or is empty.
    """
    if not os.path.exists(DEVICE_ID_FILE):
        return None

    try:
        with open(DEVICE_ID_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content if content else None
    except Exception:
        return None
