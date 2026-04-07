import logging
import queue
import threading
import time
import numpy as np

from src.core.audio.buffer import AudioChunk
from src.core.audio.stft_engine import StftEngine
from src.core.config.settings import SystemConfiguration
from src.core.utils.output_saver import OutputSaver
from src.enhancement.core.enhancement_processor import EnhancementProcessor


class EnhancementService:
    def __init__(self, config: SystemConfiguration, saver: OutputSaver, log_directory: str, dtln_model=None):
        """
        Initializes the Enhancement Service, which acts as the orchestrator for the enhancement pipeline.

        In SPEC-009 architecture, this service receives tasks from DOAService via put_task()
        and processes them in a background thread.
        """
        print("Enhancement Service Initializing...")
        self.config = config
        self.saver = saver
        self.log_dir = log_directory

        # 添加标志来表示服务是否已关闭
        self.is_closed = False

        enhancement_config = config.enhancement
        if enhancement_config is None:
            raise ValueError("Enhancement configuration section is missing in SystemConfiguration.")

        self.processor = EnhancementProcessor(config=enhancement_config, dtln_model=dtln_model)

        # --- SPEC-09: Task Queue Architecture ---
        # Queue to receive (audio_data, doa_results) tuples from DOAService
        # Large queue to prevent frame drops when enhancement processing is slower
        self.task_queue = queue.Queue(maxsize=100)
        self._running = False
        self._processing_thread = None

        # FLAC file handle for saving raw audio (set by LiveProcessingService)
        self._flac_file = None
        self._flac_lock = threading.Lock()

        # Person ID for logging (set by LiveProcessingService)
        self.person_id = None

        # Connect Components via Callbacks
        if enhancement_config.save_denoised_output:
            self.processor.add_denoise_callback(self.saver.save_denoised_chunk)
            print("EnhancementService: Denoise saving enabled.")

        if enhancement_config.dtln.enabled and enhancement_config.dtln.save_output:
            self.processor.add_dtln_callback(self.saver.save_dtln_chunk)
            print("EnhancementService: Real-time DTLN saving enabled.")
        else:
            print(f"EnhancementService: DTLN saving disabled (Enabled: {enhancement_config.dtln.enabled}, Save: {enhancement_config.dtln.save_output})")

        if enhancement_config.enable_mvdr_output_wav:
            self.processor.add_mvdr_callback(self.saver.save_mvdr_chunk)
            print("EnhancementService: Real-time MVDR saving enabled.")

        # Connect APM callback if enabled in config (check if save_apm_output exists or just assume for now)
        # We'll use a dynamic check since we haven't strictly added it to EnhancementConfig class yet,
        # but OutputSaver checks the dict.
        # However, EnhancementConfig object is passed here.
        # Let's assume if webrtc_apm is enabled, we might want to save it if OutputSaver has a file for it.
        # OutputSaver only opens the file if config says so.
        if self.saver.apm_output_file:
            self.processor.add_apm_callback(self.saver.save_apm_chunk)
            print("EnhancementService: Real-time APM output saving enabled.")

        self.processor.mvdr_logging_callback = self.saver.log_mvdr_decision

        # 角度更新超时检查
        self.angle_update_timeout = enhancement_config.angle_update_timeout if hasattr(enhancement_config, 'angle_update_timeout') else 10.0
        self.last_angle_update_time = time.time()  # 记录最后一次角度更新的时间

        # 超时标志，用于在处理线程中通知主线程关闭服务
        self.timeout_detected = False

        print("Enhancement Service Initialized and all components connected.")

    def put_task(self, audio_data: np.ndarray, doa_results: list, timestamp: float):
        """
        Thread-safe method for DOAService to submit a task to this enhancement service.

        Args:
            audio_data: Multi-channel audio data (samples, channels)
            doa_results: DOA estimation results from DOAService
            timestamp: Audio chunk timestamp
        """
        # 如果服务已超时或已关闭，则不接受新任务
        if self.timeout_detected or self.is_closed:
            return

        try:
            self.task_queue.put((audio_data, doa_results, timestamp), timeout=0.1)
        except queue.Full:
            logging.warning(f"EnhancementService task queue full, dropping frame at t={timestamp:.3f}s")

    def update_target_angle(self, angle: float):
        """
        更新目标角度并记录更新时间
        """
        self.processor.current_target_angle = angle
        self.last_angle_update_time = time.time()

    def is_angle_timeout(self) -> bool:
        """
        检查角度更新是否超时
        """
        current_time = time.time()
        return (current_time - self.last_angle_update_time) > self.angle_update_timeout

    def set_flac_file(self, flac_file):
        """Set the FLAC file handle for saving raw audio."""
        with self._flac_lock:
            self._flac_file = flac_file

    def start_processing_loop(self):
        """Start the background processing thread."""
        if self._running:
            logging.warning("EnhancementService processing loop already running")
            return
        
        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_worker, 
            name="EnhancementService-Worker",
            daemon=True
        )
        self._processing_thread.start()
        logging.info("EnhancementService processing loop started")

    def stop_processing_loop(self):
        """Stop the background processing thread gracefully."""
        if not self._running:
            return

        self._running = False

        if self._processing_thread and self._processing_thread.is_alive():
            # Put a sentinel to wake up the blocking get()
            try:
                self.task_queue.put(None, timeout=0.1)
            except queue.Full:
                pass

            # 检查是否正在从处理线程内部调用stop_processing_loop
            # 如果是，则不执行join，因为线程不能join自己
            current_thread = threading.current_thread()
            if current_thread == self._processing_thread:
                logging.debug("Stop called from processing thread itself, skipping join")
            else:
                self._processing_thread.join(timeout=2.0)

        logging.info("EnhancementService processing loop stopped")

    def _processing_worker(self):
        """Background worker that consumes tasks from the queue."""
        logging.info("EnhancementService worker thread started")

        while self._running:
            try:
                task = self.task_queue.get(timeout=0.5)

                # None is sentinel for shutdown
                if task is None:
                    break

                audio_data, doa_results, timestamp = task

                # 检查角度更新是否超时
                if self.is_angle_timeout():
                    logging.warning(f"EnhancementService for person {self.person_id} angle update timeout, stopping service...")
                    # 设置超时标志，让外部逻辑处理关闭
                    self.timeout_detected = True
                    break

                # Save raw audio to FLAC if file is set
                with self._flac_lock:
                    if self._flac_file:
                        try:
                            # Extract channels based on config
                            if self.config.recording_channels:
                                # Ensure indices are within bounds
                                valid_indices = [i for i in self.config.recording_channels if i < audio_data.shape[1]]
                                if valid_indices:
                                    save_data = audio_data[:, valid_indices]
                                else:
                                    save_data = audio_data
                            else:
                                save_data = audio_data
                            self._flac_file.write(save_data)
                        except Exception as e:
                            logging.error(f"Error writing to FLAC: {e}")

                # Create AudioChunk for processor
                chunk = AudioChunk(
                    data=audio_data,
                    num_channels=audio_data.shape[1] if audio_data.ndim > 1 else 1,
                    timestamp=timestamp
                )

                # Process the audio with timing
                import time as time_module
                start_time = time_module.perf_counter()
                self.process_audio(chunk, doa_results)
                elapsed_ms = (time_module.perf_counter() - start_time) * 1000
                if elapsed_ms > 20.0:  # Only log if processing is relatively slow (>20ms)
                    logging.debug(f"[Person {self.person_id}] Enhancement processing took {elapsed_ms:.2f}ms for {audio_data.shape[0]} samples")

            except queue.Empty:
                # 检查角度更新是否超时
                if self.is_angle_timeout():
                    logging.warning(f"EnhancementService for person {self.person_id} angle update timeout, stopping service...")
                    # 设置超时标志，让外部逻辑处理关闭
                    self.timeout_detected = True
                    # 立即退出循环，不再处理队列中的其他任务
                    break
                continue
            except Exception as e:
                logging.error(f"EnhancementService worker error: {e}", exc_info=True)

        logging.info("EnhancementService worker thread exiting")

    def process_audio(self, audio_chunk: AudioChunk, doa_results: list = None):
        """
        Processes a chunk of audio data by passing it to the enhancement processor.

        This can be called directly (legacy mode) or via the task queue (SPEC-009 mode).
        """
        # 如果服务已超时或已关闭，则不处理音频数据
        if self.timeout_detected or self.is_closed:
            return

        self.processor.process(
            audio_data=audio_chunk.data,
            timestamp=audio_chunk.timestamp,
            doa_results=doa_results
        )

        return

    def close(self):
        """Closes all downstream resources."""
        print("Enhancement Service closing...")

        # 设置关闭标志
        self.is_closed = True

        # Stop processing loop first
        self.stop_processing_loop()

        # 清空任务队列，避免在关闭后仍有任务被处理
        try:
            while True:
                self.task_queue.get_nowait()
        except queue.Empty:
            pass

        # Call processor's close method to flush any remaining DTLN buffers
        self.processor.close()

        if self.saver:
            self.saver.close()

