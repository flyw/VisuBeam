"""
This module provides the NetworkService class, which encapsulates the FastAPI application
and its API endpoints for the audio processing system.
"""
import logging
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse

# Assuming models are in a sibling file.
# The project structure is src/network/models.py
from .models import TrackingItem, LeaveNotification, AngleEnergy
from .manager import ProcessingCoordinator

"""
This module provides the NetworkService class, which encapsulates the FastAPI application
and its API endpoints for the audio processing system.
"""
import logging
import asyncio
import pathlib
import struct
import json
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# Assuming models are in a sibling file.
# The project structure is src/network/models.py
from .models import TrackingItem, LeaveNotification, AngleEnergy, DeviceIDItem, TrackingUpdateResponse
from .manager import ProcessingCoordinator

# Build path to the static directory relative to this file
STATIC_DIR = pathlib.Path(__file__).parent / "static"

class NetworkService:
    """
    Encapsulates the FastAPI application and its endpoints.
    This class is designed to be instantiated by a main application runner.
    """
    def __init__(self, coordinator: ProcessingCoordinator):
        self.app = FastAPI(
            title="DOA Audio Processing Service",
            description="Manages real-time audio enhancement based on person tracking.",
            version="0.1.0"
        )
        
        # Add CORS middleware to allow all origins
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.angle_websocket: Optional[WebSocket] = None
        self.audio_websockets: Dict[int, WebSocket] = {}
        self.coordinator = coordinator
        
        self._register_routes()
        self._register_lifecycle_events()
        self._register_middleware()

    def _register_middleware(self):
        """Register middleware."""
        from fastapi import Request
        import time
        
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            logger = logging.getLogger("live_system")
            start_time = time.time()
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000
            
            # Log only if the logger is configured (live mode)
            if logger.hasHandlers():
                logger.info(f"HTTP {request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms - {request.client.host}")
            
            return response

    async def push_angles_update(self, angles: List[AngleEnergy]):
        """
        Pushes a list of detected angles to the connected client.
        This method is intended to be called by the core audio processing logic.
        """
        if self.angle_websocket:
            try:
                # FastAPI/Pydantic v2 automatically converts models to dicts for jsonable_encoder
                await self.angle_websocket.send_json([angle.model_dump() for angle in angles])
            except WebSocketDisconnect:
                # The client disconnected.
                self.angle_websocket = None
                logging.info("Angle client disconnected during push.")
            except Exception as e:
                logging.error(f"Error pushing angle update: {e}")

    async def push_audio_chunk(self, person_id: int, chunk: bytes, metadata=None):
        """
        Pushes a chunk of enhanced audio to the client for a specific person.
        If metadata is provided, it packages it as a header.
        Format: [Header Length (4 bytes)][JSON Header][Audio Bytes]
        """
        if person_id in self.audio_websockets:
            websocket = self.audio_websockets[person_id]
            try:
                if metadata:
                    # Sanitize metadata (remove numpy arrays etc)
                    safe_metadata = []
                    if isinstance(metadata, list):
                        for frame_doa in metadata:
                            # metadata is now List[List[Tuple[float, float]]]
                            if isinstance(frame_doa, list):
                                # Ensure angles and energy are standard floats, not numpy floats
                                safe_frame = [(float(a), float(p)) for a, p in frame_doa]
                                safe_metadata.append(safe_frame)
                            else:
                                # Fallback or already sanitized
                                safe_metadata.append(frame_doa)
                    else:
                         safe_metadata = metadata

                    json_header = json.dumps(safe_metadata).encode('utf-8')
                    header_len = len(json_header)
                    # Pack: 4 bytes len, JSON bytes, Audio bytes
                    message = struct.pack(f'>I{header_len}s', header_len, json_header) + chunk
                    await websocket.send_bytes(message)
                else:
                    # Legacy behavior: just send audio bytes
                    # But wait, the client expects a format. 
                    # If we change the protocol, we should stick to it.
                    # Send empty JSON header: length 2 ("[]")
                    json_header = b'[]'
                    header_len = len(json_header)
                    message = struct.pack(f'>I{header_len}s', header_len, json_header) + chunk
                    await websocket.send_bytes(message)

            except WebSocketDisconnect:
                logging.info(f"Audio client for person_id: {person_id} disconnected during push.")
                del self.audio_websockets[person_id]
            except Exception as e:
                logging.error(f"Error pushing audio chunk for person_id {person_id}: {e}")

    def _register_lifecycle_events(self):
        """Register startup and shutdown events."""
        @self.app.on_event("startup")
        async def startup_event():
            logging.info("Application startup: Starting coordinator simulation.")
            self.coordinator.start_simulation()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            logging.info("Application shutdown: Stopping coordinator simulation.")
            self.coordinator.stop_simulation()
    
    def _register_routes(self):
        """Register the API routes with the FastAPI application."""

        @self.app.get("/", include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url="/demo.html")

        @self.app.get("/demo.html", response_class=HTMLResponse, include_in_schema=False)
        async def get_demo_page():
            demo_path = STATIC_DIR / "demo.html"
            try:
                with open(demo_path, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except FileNotFoundError:
                return HTMLResponse(content=f"<h1>Error: {demo_path} not found.</h1>", status_code=404)

        @self.app.post("/api/v1/tracking/update", response_model=List[TrackingUpdateResponse], status_code=200)
        async def tracking_update(items: List[TrackingItem]):
            """
            Updates the angles for one or more tracked individuals.
            Returns session information for each tracked person.
            """
            logging.info(f"API: Received tracking update for {len(items)} items.")
            
            # Coordinator should return list of session info dicts
            result = self.coordinator.update_tracking(items)
            
            # If coordinator returns None (e.g. simulation mode or not implemented), return empty list or mock?
            # The spec implies it should return the info.
            # LiveProcessingService now returns the list.
            if result is None:
                return []
                
            return result

        @self.app.post("/api/v1/tracking/leave", status_code=200)
        async def tracking_leave(notification: LeaveNotification):
            """
            Signals that a person has left, stopping their audio enhancement.
            """
            person_id = notification.id
            logging.info(f"API: Received leave notification for id: {person_id}")
            
            # Notify the coordinator
            if not self.coordinator.person_left(person_id):
                 raise HTTPException(status_code=404, detail=f"Person with id {person_id} not found in coordinator.")

            # Close the corresponding WebSocket connection if it exists
            if person_id in self.audio_websockets:
                websocket = self.audio_websockets[person_id]
                await websocket.close(code=1000, reason=f"Leave notification received for person {person_id}.")
                del self.audio_websockets[person_id]
                logging.info(f"Closed audio WebSocket for person_id: {person_id}")

            return {"status": "success", "message": f"Leave notification for id {person_id} processed."}

        @self.app.get("/api/v1/system/device_id", response_model=DeviceIDItem, status_code=200)
        async def get_device_id():
            """
            Retrieves the current client device ID.
            """
            from src.core.utils.device_id import load_client_device_id
            device_id = load_client_device_id()
            if device_id is None:
                raise HTTPException(status_code=404, detail="Client device ID not set.")
            return DeviceIDItem(device_id=device_id)

        @self.app.post("/api/v1/system/device_id", status_code=200)
        async def update_device_id(item: DeviceIDItem):
            """
            Updates the client device ID.
            """
            logging.info(f"API: Received device ID update: {item.device_id}")
            
            # Notify the coordinator
            if hasattr(self.coordinator, 'update_device_id'):
                self.coordinator.update_device_id(item.device_id)
            else:
                # Fallback if coordinator doesn't support it yet (or is a mock without the method)
                from src.core.utils.device_id import save_client_device_id
                save_client_device_id(item.device_id)
                logging.info(f"Saved device ID directly: {item.device_id}")

            return {"status": "success", "message": "Client device ID updated successfully."}

        @self.app.websocket("/ws/tracking/angles")
        async def ws_tracking_angles(websocket: WebSocket):
            """
            WebSocket endpoint for streaming detected angles to a single client.
            If a client is already connected, the old connection is closed to allow the new one.
            """
            await websocket.accept()
            
            if self.angle_websocket is not None:
                logging.info("New client connected. Closing existing angle stream connection.")
                try:
                    await self.angle_websocket.close(code=1000, reason="New client connected.")
                except Exception:
                    pass
                self.angle_websocket = None

            self.angle_websocket = websocket
            logging.info("Angle streaming client connected.")
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                if self.angle_websocket == websocket:
                    self.angle_websocket = None
                    logging.info("Angle streaming client disconnected.")

        @self.app.websocket("/ws/audio/enhanced/{person_id}")
        async def ws_audio_enhanced(websocket: WebSocket, person_id: int):
            """
            WebSocket endpoint for streaming enhanced audio for a specific person.
            This connection is kept alive until a 'leave' notification is received.
            """
            if person_id in self.audio_websockets:
                await websocket.close(code=1008, reason=f"An audio stream for person {person_id} is already active.")
                return

            await websocket.accept()
            self.audio_websockets[person_id] = websocket
            logging.info(f"Client connected for enhanced audio stream for person_id: {person_id}")
            try:
                # Keep the connection alive. In a real implementation, the coordinator
                # would use `push_audio_chunk` to send data.
                while True:
                    await websocket.receive_text() # Wait for disconnect
            except WebSocketDisconnect:
                logging.info(f"Client for person_id: {person_id} disconnected.")
                if person_id in self.audio_websockets:
                    del self.audio_websockets[person_id]



    def get_app(self) -> FastAPI:
        """Returns the FastAPI application instance."""
        return self.app

# Import the custom formatter for milliseconds
from src.core.utils.timestamp_formatter import MillisecondFormatter

# Configure basic logging for demonstration with custom formatter
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = MillisecondFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
