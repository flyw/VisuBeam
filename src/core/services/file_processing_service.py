#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心基础服务类
集成初始化逻辑
"""
import time
import signal
import sys
import os
import traceback
import shutil
from ..config.config_loader import load_config_from_file
from ..audio.source_selector import AudioSourceSelector
from src.core.processor.wpe_processor import WPEProcessor
from src.core.utils.output_saver import OutputSaver
from src.core.utils.log_manager import create_log_directory
from src.enhancement.core.model_loader import initialize_manager, get_manager
from src.doa.services.doa_service import DOAService
from src.enhancement.services.enhancement_service import EnhancementService
from src.core.services.processing_coordinator import ProcessingCoordinator


class FileProcessingService:
    """
    文件处理服务类
    专门负责离线 .wav 文件的处理
    """

    def __init__(self, config_path: str = "config.json", audio_file_path: str = None, output_directory: str = None):
        """
        初始化文件处理服务

        Args:
            config_path: 配置文件路径
            audio_file_path: 音频文件路径 (必须提供)
            output_directory: 输出目录路径 (可选，如果不提供则使用默认的logs/result)
        """
        if audio_file_path is None:
            raise ValueError("audio_file_path must be provided for FileProcessingService")

        self.config_path = config_path
        self.audio_file_path = audio_file_path
        self.output_directory = output_directory
        self.config = None
        self.audio_stream_pipeline = None
        self.audio_source_selector = None
        self.doa_service = None
        self.enhancement_service = None
        self.processing_coordinator = None
        self.wpe_processor = None
        self.saver = None
        self.status = "STOPPED"
        self.start_time = None
        self.error_message = None
        self.shutdown_requested = False

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 加载配置
        try:
            self.config = load_config_from_file(config_path)
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            from ..config.config_loader import get_default_config
            self.config = get_default_config()
            self.error_message = str(e)

    def _signal_handler(self, signum, frame):
        """Signal handler for graceful shutdown"""
        print(f"\nReceived signal {signum}, requesting service shutdown...")
        self.shutdown_requested = True
        if self.status == "RUNNING":
            self.stop()
        sys.exit(0)

    def initialize_system(self):
        """
        初始化系统
        """
        start_time = time.time()
        
        try:
            self.status = "STARTING"

            if not os.path.exists(self.audio_file_path):
                raise FileNotFoundError(f"Audio file does not exist: {self.audio_file_path}")

            # For file mode, save all outputs to a fixed 'result' subdirectory or custom output directory
            if self.output_directory:
                log_dir = self.output_directory
            else:
                base_log_dir = create_log_directory()  # This returns "logs"
                log_dir = os.path.join(base_log_dir, "result")

                # Clear existing 'result' directory if it exists
                if os.path.exists(log_dir):
                    print(f"Clearing existing output directory: {log_dir}")
                    shutil.rmtree(log_dir) # Remove the directory and its contents

            os.makedirs(log_dir, exist_ok=True) # Create the directory if it doesn't exist
            print(f"File mode output will be saved to: {log_dir}")
            
            # Initialize the shared OutputSaver
            self.saver = OutputSaver(
                config=self.config,
                sample_rate=self.config.sample_rate,
                num_channels=self.config.input_channels, # Will be updated if read from file? 
                # Note: input_channels in config might not match file if not set. 
                # But OutputSaver needs it. 
                # Ideally we read file info first, but OutputSaver is needed for services.
                # For now assuming config matches or we update it later.
                log_directory=log_dir
            )

            # WPE is now handled by DOAService internally
            # if self.config.wpe and self.config.wpe.enable:
            #    self.wpe_processor = WPEProcessor(...)

            # Initialize DTLN Model Manager if enabled
            dtln_model = None
            if self.config.enhancement.dtln.enabled:
                initialize_manager(self.config.enhancement.dtln)
                # For file processing, we only need 1 instance
                get_manager().preload(num_instances=1)
                dtln_model = get_manager().acquire()

            self.doa_service = DOAService(config=self.config, saver=self.saver, log_directory=log_dir)
            self.enhancement_service = EnhancementService(
                config=self.config, 
                saver=self.saver, 
                log_directory=log_dir,
                dtln_model=dtln_model
            )
            
            self.processing_coordinator = ProcessingCoordinator(self.doa_service, self.enhancement_service)

            # Initialize the source selector
            self.audio_source_selector = AudioSourceSelector(
                config=self.config,
                audio_file_path=self.audio_file_path,
                doa_service=self.processing_coordinator,
                wpe_processor=None  # WPE is handled in DOAService
            )

            # Get the pipeline
            self.audio_stream_pipeline = self.audio_source_selector.get_active_source()

            # Connect WPE save callback if enabled
            if self.config.wpe and self.config.wpe.save_output and self.wpe_processor:
                self.audio_stream_pipeline.add_wpe_callback(self.saver.save_wpe_chunk)
                print("WPE output saving is enabled.")

            self.status = "RUNNING"
            self.start_time = time.time()
            self.error_message = None

            print(f"System initialization successful, duration: {time.time() - start_time:.3f}s")
            return True

        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
            print(f"System initialization failed: {e}")
            traceback.print_exc()
            return False

    def run(self):
        """
        运行服务 (线性处理流程)
        """
        self.shutdown_requested = False
        
        if not self.initialize_system():
            print("System initialization failed, cannot run")
            sys.exit(1)

        try:
            # 启动音频流管道
            self.audio_stream_pipeline.start()
            print("File processing started...")

            while not self.shutdown_requested:
                if self.status != "RUNNING":
                    break
                
                # Exit automatically when processing is done.
                if not self.audio_stream_pipeline.is_running():
                    print("File processing finished.")
                    break

                time.sleep(0.1)

        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
            print(f"Runtime error: {e}")
            sys.exit(1)
        finally:
            if self.start_time:
                run_duration = time.time() - self.start_time
                print(f"Total runtime: {run_duration:.3f}s")

            if self.status != "STOPPED":
                self.stop()

    def stop(self):
        """Stop service"""
        self.status = "STOPPING"

        if self.processing_coordinator:
            self.processing_coordinator.close()
        elif self.doa_service:
            self.doa_service.close()

        # Release DTLN model
        if self.enhancement_service and self.config.enhancement.dtln.enabled:
             if hasattr(self.enhancement_service.processor, 'dtln_processor') and self.enhancement_service.processor.dtln_processor:
                 model = self.enhancement_service.processor.dtln_processor.model
                 try:
                     get_manager().release(model)
                 except Exception:
                     pass # Manager might not be init if failed early

        if self.audio_source_selector:
            self.audio_source_selector.stop()


        if self.audio_stream_pipeline:
            self.audio_stream_pipeline.stop()

        self.status = "STOPPED"
        print("Service stopped")

    def get_status(self):
        """获取系统状态"""
        return {
            'status': self.status,
            'start_time': self.start_time,
            'error_message': self.error_message,
            'config': self.config.to_dict() if self.config else None
        }

    def get_audio_stream(self):
        """获取音频流管道对象"""
        return self.audio_stream_pipeline

    def get_audio_device(self):
        """获取音频设备对象"""
        return self.audio_input_device

    def reload_config(self, config_path: str = None):
        """Reload configuration"""
        path = config_path or self.config_path
        try:
            new_config = load_config_from_file(path)

            # 更新服务配置
            self.config = new_config

            # 如果音频流管道存在，更新其配置
            if self.audio_stream_pipeline:
                self.audio_stream_pipeline.update_config(new_config)

            return True
        except Exception as e:
            print(f"Failed to reload configuration: {e}")
            return False
