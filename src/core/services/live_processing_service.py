import asyncio
import logging
import time
import os
import uuid
import csv
import threading
from datetime import datetime
import soundfile as sf
from typing import List, Dict, Optional, Any

import logging
from src.core.config.config_loader import load_config_from_file
from src.core.utils.log_manager import create_log_directory
from src.core.utils.output_saver import OutputSaver, NoOpOutputSaver
from src.core.audio.stream import AudioStreamPipeline
from src.core.audio.buffer import AudioChunk, SharedCircularBuffer
from src.doa.services.doa_service import DOAService
from src.enhancement.services.enhancement_service import EnhancementService
from src.network.service import NetworkService
from src.network.models import TrackingItem, AngleEnergy
from src.network.models import TrackingItem, AngleEnergy
from src.enhancement.core.model_loader import initialize_manager, get_manager

logger = logging.getLogger(__name__)


class LiveProcessingService:
    """
    Real-time processing service that integrates Audio, DOA, Enhancement, and Network modules.
    Acts as the coordinator for the NetworkService.
    """

    def __init__(self, config_path: str = "config.json"):
        # Configure GPU memory growth to prevent TF from hogging all VRAM
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Enabled GPU memory growth for {len(gpus)} devices.")
        except Exception as e:
            logging.warning(f"Failed to configure GPU memory growth: {e}")

        self.config_path = config_path
        self.config = load_config_from_file(config_path)
        
        # Initialize DTLN Model Manager and preload models
        if self.config.enhancement.dtln.enabled:
            initialize_manager(self.config.enhancement.dtln)
            # Preload instances based on config.pool_size (defaulted in initialize_manager)
            get_manager().preload(num_instances=self.config.enhancement.dtln.pool_size)
        
        # Initialize NetworkService with self as coordinator
        self.network_service = NetworkService(coordinator=self)
        self.app = self.network_service.get_app()
        self.loop = None

        
        # Shared resources
        self.log_dir = create_log_directory()
        self._state_lock = threading.Lock()
        
        # Service Monitoring
        self._monitor_running = False
        self._monitor_thread = None

        # Use OutputSaver instead of NoOpOutputSaver to enable saving of mcra output
        self.saver = OutputSaver(
            config=self.config,
            sample_rate=self.config.sample_rate,
            num_channels=self.config.input_channels or 0,
            log_directory=self.log_dir
        )

        # Disable plotting but keep MCRA output enabled if configured
        if self.config.doa:
            self.config.doa.realtime_plot_enabled = False
            self.config.doa.global_heatmap_enabled = False
            self.config.doa.global_doa_plot_enabled = False
            # TODO MCRA output is disable by default
            self.config.doa.save_mcra_output = False
            logging.info("Disabled DOA plotting/heatmap.")
        
        # Global DOA Service
        self.doa_service = DOAService(config=self.config, saver=self.saver, log_directory=self.log_dir)
        
        # Register DOA result callback for pushing to WebSocket
        self.doa_service.doa_result_callbacks.append(self._on_doa_results)

        # Enhancement Services (Dynamic)
        self.enhancement_services: Dict[int, EnhancementService] = {}
        self.tracked_persons: Dict[int, float] = {} # ID -> Angle
        # Removed ThreadPoolExecutor - SPEC-009 uses EnhancementService's own processing thread
        
        # Session Management
        # person_id -> { 'uuid': str, 'dir': str, 'flac_file': sf.SoundFile, 'angle_log': file_handle }
        self.sessions: Dict[int, Dict[str, Any]] = {}

        # --- SPEC-009: Shared Buffer Architecture ---
        # Calculate shared buffer capacity based on config
        num_channels = self.config.input_channels or len(self.config.mic_positions) or 4
        buffer_samples = int(self.config.shared_buffer_duration_ms / 1000 * self.config.sample_rate)
        self.shared_buffer = SharedCircularBuffer(
            capacity_samples=buffer_samples,
            num_channels=num_channels
        )
        logging.info(f"Created SharedCircularBuffer: {buffer_samples} samples, {num_channels} channels")

        # Audio Pipeline - now uses shared_buffer instead of direct callback
        self.audio_pipeline = AudioStreamPipeline(
            config=self.config,
            audio_file_path=None,  # Real-time mode
            shared_buffer=self.shared_buffer,  # SPEC-009: write to shared buffer
            wpe_processor=None
        )
        
        # WPE is now handled within DOAService -> DOAProcessor to avoid blocking I/O thread.
        # So we don't initialize it here or attach it to audio_pipeline.

        self._register_lifecycle_events()

    def _register_lifecycle_events(self):
        @self.app.on_event("startup")
        async def startup_event():
            logging.info("LiveProcessingService starting up...")
            self.loop = asyncio.get_running_loop()
            self.start()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            logging.info("LiveProcessingService shutting down...")
            self.stop()

    def _monitor_services_loop(self):
        """
        Background thread to monitor services for timeouts and perform cleanup.
        Required because process_audio is not called in SPEC-009 architecture.
        """
        logging.info("Service monitoring loop started")
        while self._monitor_running:
            try:
                # 检查并清理已关闭或超时的EnhancementService
                closed_services = []
                with self._state_lock:
                    services_snapshot = list(self.enhancement_services.items())
                
                for person_id, service in services_snapshot:
                    if service.is_closed or service.timeout_detected:
                        closed_services.append(person_id)

                # 从字典中移除已关闭的服务
                for person_id in closed_services:
                    logging.info(f"Monitor detected timeout/closure for person {person_id}, cleaning up...")
                    # 直接调用person_left方法，复用完整的清理流程
                    self.person_left(person_id)
                
                time.sleep(1.0)  # Check every second
            except Exception as e:
                logging.error(f"Error in service monitoring loop: {e}")
                time.sleep(1.0)
        logging.info("Service monitoring loop stopped")

    def start(self):
        """Start the audio pipeline and DOA processing loop."""
        logging.info("Starting audio pipeline...")
        logging.info("Opening microphone array...")
        self.audio_pipeline.start()
        
        # SPEC-009: Start DOAService processing loop (reads from shared buffer)
        logging.info("Starting DOAService processing loop...")
        self.doa_service.start_processing_loop(self.shared_buffer)
        
        # Start Service Monitor
        if not self._monitor_running:
            self._monitor_running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_services_loop,
                name="LiveProcessingService-Monitor",
                daemon=True
            )
            self._monitor_thread.start()

    def stop(self):
        """Stop the audio pipeline and close resources."""
        logging.info("Stopping audio pipeline...")
        self.audio_pipeline.stop()

        # Stop Service Monitor
        self._monitor_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        # SPEC-009: DOAService close will stop its processing loop
        self.doa_service.close()
        
        # Close all enhancement services (their close() will stop their processing loops)
        with self._state_lock:
            for enhancer in self.enhancement_services.values():
                # Release DTLN model back to pool
                if hasattr(enhancer.processor, 'dtln_processor') and enhancer.processor.dtln_processor:
                    model = enhancer.processor.dtln_processor.model
                    if self.config.enhancement.dtln.enabled:
                        get_manager().release(model)
                enhancer.close()
        
        self.saver.close()

    def get_fastapi_app(self):
        return self.app

    # --- Coordinator Interface Implementation ---

    def update_tracking(self, items: List[TrackingItem]) -> List[Dict[str, Any]]:
        """
        Called by NetworkService when tracking info is updated.
        Returns session info for each item.
        """
        response_data = []
        current_time = datetime.now()
        # Format date and time with hyphens and underscores
        formatted_datetime = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        # Add milliseconds
        timestamp_str = f"{formatted_datetime}_{current_time.microsecond // 1000:03d}"
        
        with self._state_lock:
            for item in items:
                # Check for existing angle
                current_angle = self.tracked_persons.get(item.id)
                session_exists = item.id in self.sessions

                # Update logic is needed if:
                # 1. Session doesn't exist (need to create it)
                # 2. Angle changed (strict equality since minimal unit is 1 degree)
                angle_changed = current_angle is None or item.angle != current_angle
                is_update_needed = not session_exists or angle_changed
                
                if is_update_needed:
                    self.tracked_persons[item.id] = item.angle
                
                    # 1. Manage Session
                    if not session_exists:
                        # Create new session
                        session_uuid = str(uuid.uuid4())
                        dir_name = f"{timestamp_str}-{item.id}-{session_uuid}"
                        session_dir = os.path.join(self.log_dir, dir_name)
                        os.makedirs(session_dir, exist_ok=True)
                        
                        # Initialize FLAC saver
                        flac_path = os.path.join(session_dir, "raw_mic_audio.flac")
                        # We save channels defined in config or all input channels
                        if self.config.recording_channels:
                            channels = len(self.config.recording_channels)
                        else:
                            channels = self.config.input_channels or 4
                        samplerate = self.config.sample_rate
                        
                        try:
                            flac_file = sf.SoundFile(flac_path, mode='w', samplerate=samplerate, channels=channels, subtype='PCM_24')
                        except Exception as e:
                            logging.error(f"Failed to create FLAC file for person {item.id}: {e}")
                            flac_file = None

                        # Initialize Angle Log
                        angle_log_path = os.path.join(session_dir, "received_angle_history.log")
                        angle_log_file = open(angle_log_path, "a", encoding="utf-8")
                        
                        self.sessions[item.id] = {
                            'uuid': session_uuid,
                            'dir_name': dir_name,
                            'dir_path': session_dir,
                            'flac_file': flac_file,
                            'angle_log': angle_log_file
                        }
                        logging.info(f"Started new session for person {item.id}: {session_uuid}")

                    session = self.sessions[item.id]
                    
                    # 2. Log Angle
                    iso_time = datetime.now().isoformat(timespec='milliseconds') + "Z"
                    if session['angle_log']:
                        session['angle_log'].write(f"{iso_time}, {item.angle}\n")
                        session['angle_log'].flush()
                    
                    # 3. Create/Update EnhancementService (Existing logic)
                    if item.id not in self.enhancement_services:
                        logging.info(f"Creating new EnhancementService for person {item.id}")

                        # Acquire DTLN model from pool
                        dtln_model = None
                        if self.config.enhancement.dtln.enabled:
                            try:
                                dtln_model = get_manager().acquire()
                                logging.info(f"Acquired DTLN model for person {item.id}")
                            except Exception as e:
                                logging.error(f"Failed to acquire DTLN model: {e}")
                                pass

                        # For live mode, create a dedicated OutputSaver for each person to avoid mixing audio
                        # when saving DTLN output
                        session_dir = self.sessions[item.id]['dir_path']
                        person_saver = OutputSaver(
                            config=self.config,
                            sample_rate=self.config.sample_rate,
                            num_channels=1,  # DTLN output is mono
                            log_directory=session_dir
                        )

                        service = EnhancementService(
                            config=self.config,
                            saver=person_saver,
                            log_directory=session_dir,
                            dtln_model=dtln_model
                        )
                        
                        # Set FLAC file for raw audio saving
                        flac_file = self.sessions[item.id].get('flac_file')
                        if flac_file:
                            service.set_flac_file(flac_file)
                        
                        # Set person ID for logging
                        service.person_id = item.id
                        
                        # Register callback to push enhanced audio
                        def make_callback(pid):
                            return lambda chunk, metadata=None: self._on_enhanced_audio(pid, chunk, metadata)
                        
                        network_callback = make_callback(item.id)
                        
                        # Register callback based on enabled features (DTLN > MVDR)
                        if self.config.enhancement.dtln.enabled:
                            service.processor.add_dtln_callback(network_callback)
                            logging.info(f"Registered DTLN callback for person {item.id} to network")
                        elif self.config.enhancement.enable_mvdr_output_wav:
                            service.processor.add_mvdr_callback(network_callback)
                            logging.info(f"Registered MVDR callback for person {item.id} to network")
                        else:
                            logging.warning(f"No suitable output stage enabled for person {item.id} network stream")
                        
                        self.enhancement_services[item.id] = service
                        
                        # SPEC-009: Start the service's processing loop and register as consumer
                        service.start_processing_loop()
                        self.doa_service.register_consumer(service)
                        logging.info(f"Started processing and registered consumer for person {item.id}")

                else:
                    # Update not needed, just grab session for response
                    session = self.sessions[item.id]

                # ALWAYS update the target angle to reset the timeout timer, 
                # even if the angle hasn't changed. This acts as a heartbeat.
                if item.id in self.enhancement_services:
                    self.enhancement_services[item.id].update_target_angle(item.angle)
                    if is_update_needed:
                         logging.debug(f"Updated target angle for person {item.id} to {item.angle}")

                response_data.append({
                    "id": item.id,
                    "session_uuid": session['uuid'],
                    "log_directory_name": session['dir_name']
                })

        return response_data

    def person_left(self, person_id: int) -> bool:
        """
        Called by NetworkService when a person leaves.
        """
        found = False
        
        with self._state_lock:
            # 1. Close Session Resources
            if person_id in self.sessions:
                logging.info(f"Closing session for person {person_id}")
                session = self.sessions[person_id]
                
                # Close FLAC file
                if session.get('flac_file'):
                    try:
                        session['flac_file'].close()
                    except Exception as e:
                        logging.error(f"Error closing FLAC file: {e}")
                
                # Close Angle Log
                if session.get('angle_log'):
                    try:
                        session['angle_log'].close()
                    except Exception as e:
                        logging.error(f"Error closing angle log: {e}")

                del self.sessions[person_id]
                found = True

            # 2. Remove Enhancement Service
            if person_id in self.enhancement_services:
                logging.info(f"Removing EnhancementService for person {person_id}")
                service = self.enhancement_services[person_id]
                
                # SPEC-009: Unregister from DOAService first
                self.doa_service.unregister_consumer(service)

                # Release DTLN model back to pool
                if self.config.enhancement.dtln.enabled:
                    if hasattr(service.processor, 'dtln_processor') and service.processor.dtln_processor:
                        model = service.processor.dtln_processor.model
                        get_manager().release(model)
                        logging.info(f"Released DTLN model for person {person_id}")

                # close() will stop the processing loop
                service.close()

                del self.enhancement_services[person_id]
                found = True
                
            # 3. Remove from tracked persons
            if person_id in self.tracked_persons:
                del self.tracked_persons[person_id]
                found = True
            
        return found

    def start_simulation(self):
        # No-op, we use real audio
        pass

    def stop_simulation(self):
        # No-op
        pass

    def update_device_id(self, device_id: str):
        """
        Updates the client device ID.
        """
        logging.info(f"Updating client device ID to: {device_id}")
        from src.core.utils.device_id import save_client_device_id
        save_client_device_id(device_id)

    # --- Audio Processing Loop ---

    def process_audio(self, audio_chunk: AudioChunk):
        """
        Callback from AudioStreamPipeline.
        """
        logger.debug(
            f"LPS - Received AudioChunk: id={audio_chunk.id}, "
            f"timestamp={audio_chunk.timestamp:.4f}s, "
            f"data_shape={audio_chunk.data.shape}"
        )

        # 0. Save Raw Audio to Sessions
        # We do this first to ensure raw data is captured
        # audio_chunk.data is numpy array (frames, channels)
        
        # Snapshot sessions to avoid iteration error
        with self._state_lock:
            sessions_snapshot = list(self.sessions.values())

        if sessions_snapshot:
            # Extract channels based on config
            if self.config.recording_channels:
                 # Ensure indices are within bounds
                 valid_indices = [i for i in self.config.recording_channels if i < audio_chunk.data.shape[1]]
                 if valid_indices:
                     save_data = audio_chunk.data[:, valid_indices]
                 else:
                     save_data = audio_chunk.data
            else:
                 save_data = audio_chunk.data

            for session in sessions_snapshot:
                if session.get('flac_file'):
                    try:
                        # Ensure data is in correct format for soundfile (usually float32 or int16)
                        # AudioChunk usually has float32 data in range [-1, 1]
                        session['flac_file'].write(save_data)
                    except Exception as e:
                        logging.error(f"Error writing to FLAC for session {session['uuid']}: {e}")

        # 1. Global DOA
        doa_results, _ = self.doa_service.process_audio(audio_chunk)
        
        # 2. Push DOA to Network
        if self.loop and doa_results:
            # Convert doa_results to AngleEnergy list
            # doa_results is a list of dicts? Or list of tuples?
            # DOAService returns: [{'timestamp': t, 'doa': [(angle, prob), ...], 'spatial_spectrum': ...}]
            # We need to extract the last result
            latest_result = doa_results[-1]
            if 'doa' in latest_result:
                angles_list = []
                for angle, energy in latest_result['doa']:
                    angles_list.append(AngleEnergy(angle=angle, energy=energy))
                
                if angles_list:
                    asyncio.run_coroutine_threadsafe(
                        self.network_service.push_angles_update(angles_list),
                        self.loop
                    )

        # 3. Enhancement for each tracked person
        # Snapshot services and tracked persons
        with self._state_lock:
            services_snapshot = list(self.enhancement_services.items())
            tracked_persons_snapshot = self.tracked_persons.copy()

        # 检查并清理已关闭或超时的EnhancementService
        closed_services = []
        for person_id, service in services_snapshot:
            if service.is_closed or service.timeout_detected:
                closed_services.append(person_id)

        # 从字典中移除已关闭的服务
        with self._state_lock:
            for person_id in closed_services:
                # 直接调用person_left方法，复用完整的清理流程
                self.person_left(person_id)

        # 重新获取快照，因为可能已移除一些服务
        with self._state_lock:
            services_snapshot = list(self.enhancement_services.items())

        for person_id, service in services_snapshot:
            # 检查服务是否已超时或已关闭，如果是则跳过
            if service.is_closed or service.timeout_detected:
                continue

            target_angle = tracked_persons_snapshot.get(person_id)
            if target_angle is not None:
                # Update the target angle for the enhancement service. This is thread-safe
                # as it's just overwriting a float.
                service.update_target_angle(target_angle)

                # Create thread-safe copies of data for the background thread.
                doa_results_copy = [dict(result) for result in doa_results] if doa_results else []
                chunk_copy = audio_chunk.copy()

                # Submit the heavy enhancement task to the service's own processing thread
                # to avoid blocking the main audio processing loop.
                try:
                    service.put_task(
                        chunk_copy.data,
                        doa_results_copy,
                        chunk_copy.timestamp
                    )
                except Exception as e:
                    logging.error(f"Failed to submit task to EnhancementService for person {person_id}: {e}")

    def _on_doa_results(self, doa_results):
        """
        Callback invoked by DOAService when DOA results are available.
        Pushes angles to WebSocket clients.
        """
        if not self.loop:
            logging.warning("_on_doa_results: No event loop available")
            return
        if not doa_results:
            return
        
        try:
            # doa_results is a list of dicts: [{'timestamp': t, 'doa': [(angle, prob), ...], ...}]
            latest_result = doa_results[-1] if doa_results else None
            if latest_result and 'doa' in latest_result:
                angles_list = []
                for angle, energy in latest_result['doa']:
                    angles_list.append(AngleEnergy(angle=angle, energy=energy))
                
                if angles_list:
                    asyncio.run_coroutine_threadsafe(
                        self.network_service.push_angles_update(angles_list),
                        self.loop
                    )
        except Exception as e:
            logging.error(f"Error in _on_doa_results: {e}", exc_info=True)

    def _on_enhanced_audio(self, person_id: int, chunk_data, metadata=None):
        """
        Callback for enhanced audio.
        chunk_data is likely np.ndarray or bytes?
        NetworkService expects bytes.
        """
        logging.debug(f"_on_enhanced_audio called for person {person_id}, chunk type: {type(chunk_data)}, loop: {self.loop is not None}")
        if self.loop:
            # Convert numpy to bytes if needed
            import numpy as np
            if isinstance(chunk_data, np.ndarray):
                # logging.debug(f"Converting numpy array to bytes: shape={chunk_data.shape}, dtype={chunk_data.dtype}")
                # Ensure int16
                if chunk_data.dtype != np.int16:
                     chunk_data = (np.clip(chunk_data, -1.0, 1.0) * 32767.0).astype(np.int16)
                data_bytes = chunk_data.tobytes()
            else:
                data_bytes = chunk_data
            
            asyncio.run_coroutine_threadsafe(
                self.network_service.push_audio_chunk(person_id, data_bytes, metadata),
                self.loop
            )
