import logging
import threading
import time
from typing import List, Optional, TYPE_CHECKING

from src.core.audio.buffer import AudioChunk
from src.core.config.settings import SystemConfiguration
from src.core.utils.output_saver import OutputSaver
from src.doa.core.doa_processor import DOAProcessor
from src.doa.config.doa_config import DOAConfig
from src.doa.visualization.global_visualizer import GlobalVisualizer
from src.doa.visualization.realtime_visualizer import RealtimeVisualizer

if TYPE_CHECKING:
    from src.core.audio.buffer import SharedCircularBuffer
    from src.enhancement.services.enhancement_service import EnhancementService


class DOAService:
    def __init__(self, config: SystemConfiguration, saver: OutputSaver, log_directory: str):
        """
        Initializes the DOA Service, which orchestrates the entire DOA pipeline.
        
        In SPEC-009 architecture, DOAService acts as:
        - Primary consumer from SharedCircularBuffer
        - Fan-out hub distributing (audio_data, doa_results) to EnhancementService consumers
        """
        print("DOA Service Initializing...")
        self.config = config
        self.saver = saver
        self.log_dir = log_directory
        
        doa_config = config.doa
        wpe_config = config.wpe
        if doa_config is None:
            raise ValueError("DOA configuration section is missing in SystemConfiguration.")

        self.processor = DOAProcessor(config=doa_config, wpe_config=wpe_config)

        # --- SPEC-009: Consumer Registration for Fan-out ---
        self.enhancement_consumers: List['EnhancementService'] = []
        self._consumers_lock = threading.Lock()
        
        # --- SPEC-009: Processing Loop State ---
        self._shared_buffer: Optional['SharedCircularBuffer'] = None
        self._running = False
        self._processing_thread = None
        
        # Calculate frame size for processing (samples per frame)
        self._frame_length_samples = int(doa_config.frame_length_ms / 1000 * doa_config.sample_rate)
        
        # --- DOA Result Callbacks (for pushing to network, etc.) ---
        self.doa_result_callbacks = []
        
        self.original_audio_callbacks = []

        self.realtime_visualizer = None
        if doa_config.realtime_plot_enabled:
            self.realtime_visualizer = RealtimeVisualizer(
                config=doa_config,
                sample_rate=doa_config.sample_rate,
                num_channels=doa_config.num_mics,
                log_directory=self.log_dir
            )

        self.global_visualizer = None
        if doa_config.global_heatmap_enabled or doa_config.global_doa_plot_enabled:
            self.global_visualizer = GlobalVisualizer(
                config=doa_config,
                sample_rate=doa_config.sample_rate,
                num_channels=doa_config.num_mics,
                log_directory=self.log_dir
            )
            print("GlobalVisualizer created and initialized.")

        # Connect Components via Callbacks
        if config.wpe and config.wpe.enable and config.wpe.save_output:
             self.processor.add_wpe_callback(self.saver.save_wpe_chunk)

        if doa_config.save_mcra_output:
            self.processor.add_mcra_callback(self.saver.save_mcra_chunk)
        if doa_config.save_original_audio:
            self.add_original_audio_callback(self.saver.save_original_chunk)

        if self.realtime_visualizer:
            self.processor.add_realtime_plot_callback(self.realtime_visualizer.update_plot_and_save)

        if self.global_visualizer:
            self.processor.add_realtime_plot_callback(self.global_visualizer.accumulate_data)

        self.processor.doa_logging_callback = self.saver.log_doa_result

        print("DOA Service Initialized and all components connected.")

    # --- SPEC-009: Consumer Management ---
    
    def register_consumer(self, service: 'EnhancementService'):
        """Register an EnhancementService to receive fan-out data."""
        with self._consumers_lock:
            if service not in self.enhancement_consumers:
                self.enhancement_consumers.append(service)
                logging.info(f"DOAService: Registered consumer, total: {len(self.enhancement_consumers)}")

    def unregister_consumer(self, service: 'EnhancementService'):
        """Unregister an EnhancementService from receiving data."""
        with self._consumers_lock:
            if service in self.enhancement_consumers:
                self.enhancement_consumers.remove(service)
                logging.info(f"DOAService: Unregistered consumer, total: {len(self.enhancement_consumers)}")

    # --- SPEC-009: Shared Buffer Processing Loop ---
    
    def start_processing_loop(self, shared_buffer: 'SharedCircularBuffer'):
        """
        Start the main processing loop that reads from SharedCircularBuffer.
        
        This makes DOAService the primary consumer of audio data, processing DOA
        and then fan-out distributing to all registered EnhancementService consumers.
        """
        if self._running:
            logging.warning("DOAService processing loop already running")
            return
        
        self._shared_buffer = shared_buffer
        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_worker,
            name="DOAService-Worker",
            daemon=True
        )
        self._processing_thread.start()
        print(f"DOAService processing loop started, frame_length={self._frame_length_samples} samples")

    def stop_processing_loop(self):
        """Stop the processing loop gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        logging.info("DOAService processing loop stopped")

    def _processing_worker(self):
        """Background worker that reads from SharedCircularBuffer and fans out."""
        logging.info("DOAService worker thread started")
        
        # Wait a bit for buffer to accumulate initial data
        time.sleep(0.1)
        
        while self._running:
            try:
                # Read frame using consuming read (moves read pointer, no data loss)
                audio_data = self._shared_buffer.read_consume(self._frame_length_samples)
                
                if audio_data is None:
                    # Not enough data yet, wait a bit
                    time.sleep(0.01)
                    continue
                
                # Create AudioChunk
                timestamp = time.time()
                chunk = AudioChunk(
                    data=audio_data,
                    num_channels=audio_data.shape[1] if audio_data.ndim > 1 else 1,
                    timestamp=timestamp
                )
                
                # Process DOA and get WPE/MCRA processed audio
                doa_results, processed_audio = self.process_audio(chunk)
                
                # Fan-out to all registered consumers
                # Use raw audio_data instead of processed_audio to ensure ref channel is available for AEC
                with self._consumers_lock:
                    for consumer in self.enhancement_consumers:
                        try:
                            # Pass original audio_data to ensure reference channels are available for AEC
                            consumer.put_task(audio_data, doa_results, timestamp)
                        except Exception as e:
                            logging.error(f"Error distributing to consumer: {e}")

                # Invoke DOA result callbacks (e.g., for pushing to network)
                for callback in self.doa_result_callbacks:
                    try:
                        callback(doa_results)
                    except Exception as e:
                        logging.error(f"Error in DOA result callback: {e}")
                
                # No sleep needed - read_consume blocks until data is available
                # Processing is now driven by data arrival rate
                
            except Exception as e:
                logging.error(f"DOAService worker error: {e}", exc_info=True)
                time.sleep(0.1)
        
        logging.info("DOAService worker thread exiting")

    # --- Original Methods ---

    def process_audio(self, audio_chunk: AudioChunk):
        """
        Processes a chunk of audio data by passing it to the DOA processor.
        
        Can be called directly (legacy/file mode) or via processing loop (SPEC-009 mode).
        Returns:
            tuple: (doa_results, processed_data)
        """
        self._invoke_original_audio_callbacks(audio_chunk)

        doa_results, processed_data = self.processor.process(
            audio_data=audio_chunk.data,
            timestamp=audio_chunk.timestamp
        )

        return doa_results, processed_data

    def _invoke_original_audio_callbacks(self, audio_chunk):
        for callback in self.original_audio_callbacks:
            try:
                callback(audio_chunk.data)
            except Exception as e:
                print(f"Error in original audio callback: {e}")

    def add_original_audio_callback(self, callback):
        self.original_audio_callbacks.append(callback)

    def close(self):
        """Closes all downstream resources."""
        print("DOA Service closing...")
        
        # Stop processing loop first
        self.stop_processing_loop()

        # Call processor's close method
        self.processor.close()

        if self.saver:
            self.saver.close()
        if self.realtime_visualizer:
            self.realtime_visualizer.close()
        if self.global_visualizer:
            self.global_visualizer.close()

