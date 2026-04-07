#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理系统核心基础模块 - 主入口点

支持命令行参数：
--config: 配置文件路径
--audio-file: 音频文件路径（用于处理预录制音频）
"""
import argparse
import sys
import json
import time
from src.core.services.file_processing_service import FileProcessingService
from src.core.services.live_processing_service import LiveProcessingService
from src.core.config.config_loader import load_config_from_file, save_config_to_file
from src.core.config.settings import SystemConfiguration
from src.core.monitoring import SystemMonitor
import logging
import uvicorn

# 全局服务实例，用于在不同命令间共享
current_service = None




# 全局服务实例，用于在不同命令间共享
current_service = None

# Configure TensorFlow memory growth to prevent it from hogging all GPU memory
# This is crucial when co-running with PyTorch on the same GPU
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"TensorFlow memory growth enabled for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.error(f"Failed to set TensorFlow memory growth: {e}")
except ImportError:
    logging.warning("TensorFlow not installed or import failed. Skipping memory configuration.")


def main():
    """主函数，处理命令行参数并启动服务"""
    parser = argparse.ArgumentParser(description="音频处理系统核心基础模块")

    # 定义可用命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # run 命令
    run_parser = subparsers.add_parser('run', help='运行音频处理系统')
    run_parser.add_argument("--config", type=str, default="config.yaml",
                           help="Configuration file path (default: config.yaml)")
    run_parser.add_argument("--audio-file", type=str,
                           help="音频文件路径 (用于文件处理模式)")
    run_parser.add_argument("--daemon", action="store_true",
                           help="以守护进程模式运行 (仅文件处理模式)")
    run_parser.add_argument("--host", type=str, default="0.0.0.0",
                           help="服务绑定的主机地址 (仅服务模式)")
    run_parser.add_argument("--port", type=int, default=8000,
                           help="服务绑定的端口 (仅服务模式)")
    
    # status 命令
    status_parser = subparsers.add_parser('status', help='Display system runtime status')
    status_parser.add_argument("--config", type=str, default="config.yaml",
                              help="Configuration file path (default: config.yaml)")
    status_parser.add_argument("--format", type=str, choices=['json', 'table', 'text'], 
                              default='table', help="Output format (default: table)")
    status_parser.add_argument("--watch", action="store_true",
                              help="Continuous monitoring mode")
    
    # config 命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_subparsers = config_parser.add_subparsers(dest='config_cmd', help='配置子命令')
    
    # config get
    config_get_parser = config_subparsers.add_parser('get', help='获取当前配置')
    config_get_parser.add_argument("--config", type=str, default="config.yaml",
                                  help="Configuration file path (default: config.yaml)")
    
    # config list-devices
    config_list_parser = config_subparsers.add_parser('list-devices', help='列出可用音频设备')
    
    # config validate
    config_validate_parser = config_subparsers.add_parser('validate', help='验证配置文件')
    config_validate_parser.add_argument("--config", type=str, default="config.yaml",
                                      help="Configuration file path (default: config.yaml)")
    
    # monitor 命令
    monitor_parser = subparsers.add_parser('monitor', help='实时监控系统性能指标')
    monitor_parser.add_argument("--interval", type=float, default=1.0,
                               help="更新间隔 (秒) (默认: 1)")
    monitor_parser.add_argument("--duration", type=float,
                               help="监控持续时间 (默认: 无限)")
    monitor_parser.add_argument("--format", type=str, choices=['table', 'json'],
                               default='table', help="输出格式 (默认: table)")

    # stop 命令
    stop_parser = subparsers.add_parser('stop', help='停止运行的音频处理服务')
    stop_parser.add_argument("--force", action="store_true", 
                            help="强制停止")
    stop_parser.add_argument("--timeout", type=int, default=10,
                            help="等待超时时间 (默认: 10)")
    
    # record-ref 命令
    record_parser = subparsers.add_parser('record-ref', help='录制参考通道音频以进行测试')
    record_parser.add_argument("--config", type=str, default="config.yaml",
                              help="配置文件路径 (默认: config.yaml)")
    record_parser.add_argument("--output", type=str, default="reference_recording.wav",
                              help="输出文件名 (默认: reference_recording.wav)")

    # 解析参数
    args = parser.parse_args()

    # 如果没有提供命令，默认为run
    if not args.command:
        args.command = 'run'
        # 设置默认参数
        if not hasattr(args, 'config'):
            args.config = 'config.yaml'
        if not hasattr(args, 'audio_file'):
            args.audio_file = None
        if not hasattr(args, 'daemon'):
            args.daemon = False
        if not hasattr(args, 'host'):
            args.host = '0.0.0.0'
        if not hasattr(args, 'port'):
            args.port = 8000

    # 根据命令执行相应功能
    if args.command == "run":
        # 如果指定了 audio-file，则进入文件处理模式
        if args.audio_file:
            print(f"Starting in file-processing mode for: {args.audio_file}")
            
            # Pre-load the model to avoid delay on first use
            from src.core.config.config_loader import load_config_from_file
            from src.enhancement.core.model_loader import get_manager, initialize_manager
            print("Loading configuration for model pre-loading...")
            config = load_config_from_file(args.config)
            
            # Initialize and preload DTLN model manager if enabled
            if config.enhancement.dtln.enabled:
                initialize_manager(config.enhancement.dtln)
                get_manager().preload(num_instances=1)
            
            service = FileProcessingService(config_path=args.config, audio_file_path=args.audio_file)
            global current_service
            current_service = service
            service.run()
        # 否则，进入实时服务模式
        else:
            print("Starting in live-service mode with network API...")
            
            # Setup global logging to capture stdout, stderr, and exceptions
            from src.core.utils.log_manager import setup_global_logging
            logger = setup_global_logging()
            logger.setLevel(logging.DEBUG) # Set logging level to DEBUG
            logger.info("System starting up in live mode with global logging.")
            
            # Load client device ID
            from src.core.utils.device_id import load_client_device_id
            device_id = load_client_device_id()
            if device_id:
                logger.info(f"Loaded client device ID: {device_id}")
            else:
                logger.info("No client device ID found.")

            live_service = LiveProcessingService(config_path=args.config)
            app = live_service.get_fastapi_app()
            print(f"Web service running on http://{args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "status":
        handle_status_command(args)
    elif args.command == "config":
        handle_config_command(args)
    elif args.command == "monitor":
        handle_monitor_command(args)
    elif args.command == "stop":
        handle_stop_command(args)
    elif args.command == "record-ref":
        from src.core.utils.audio_recorder import record_reference_standalone
        record_reference_standalone(config_path=args.config, output_file=args.output)
    else:
        print(f"未知命令: {args.command}")
        parser.print_help()
        sys.exit(1)


def handle_status_command(args):
    """处理 status 命令"""
    global current_service
    
    # 尝试获取当前运行的服务状态，或者创建一个临时服务以获取配置信息
    if current_service:
        status = current_service.get_status()
    else:
        # 如果没有运行的服务，至少显示配置信息
        try:
            config = load_config_from_file(args.config)
            status = {
                'status': 'STOPPED',
                'config': config.to_dict(),
                'message': 'Service not running, displaying config information'
            }
        except Exception as e:
            status = {
                'status': 'ERROR',
                'error': str(e),
                'message': 'Cannot load config file'
            }
    
    if args.format == 'json':
        print(json.dumps(status, indent=2, ensure_ascii=False))
    elif args.format == 'table':
        print_status_table(status)
    else:  # text format
        print_status_text(status)
    
    # 如果是监控模式，则定期更新状态
    if args.watch:
        try:
            while True:
                time.sleep(1)
                if current_service:
                    status = current_service.get_status()
                else:
                    config = load_config_from_file(args.config)
                    status = {
                        'status': 'STOPPED',
                        'config': config.to_dict(),
                        'message': 'Service not running'
                    }
                
                print("\033[2J\033[H", end='')  # 清屏
                if args.format == 'json':
                    print(json.dumps(status, indent=2, ensure_ascii=False))
                elif args.format == 'table':
                    print_status_table(status)
                else:
                    print_status_text(status)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")


def print_status_table(status_info):
    """Print status in table format"""
    print("System Status Information")
    print("-" * 30)
    for key, value in status_info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


def print_status_text(status_info):
    """以文本格式打印状态信息"""
    for key, value in status_info.items():
        print(f"{key}: {value}")


def handle_config_command(args):
    """Handle config command"""
    if not args.config_cmd:
        print("Please specify config sub-command: get, list-devices, validate")
        return
    
    if args.config_cmd == "get":
        try:
            config = load_config_from_file(args.config)
            print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: Cannot load config file {args.config} - {e}")
    
    elif args.config_cmd == "list-devices":
        list_audio_devices()
    
    elif args.config_cmd == "validate":
        try:
            config = load_config_from_file(args.config)
            config.validate()
            print(f"Config file {args.config} validation successful")
        except Exception as e:
            print(f"Config file {args.config} validation failed: {e}")


def list_audio_devices():
    """List available audio devices"""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        print(f"{'Index':<5} {'Name':<50} {'Max Input Ch':<10} {'Max Output Ch':<10} {'Default Rate':<15}")
        print("-" * 90)
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"{info['index']:<5} {info['name'][:49]:<50} {info['maxInputChannels']:<10} "
                  f"{info['maxOutputChannels']:<10} {int(info['defaultSampleRate']):<15}")
        
        p.terminate()
        
    except ImportError:
        print("PyAudio not installed, cannot list audio devices")
    except Exception as e:
        print(f"Error listing audio devices: {e}")


def handle_monitor_command(args):
    """Handle monitor command"""
    monitor = SystemMonitor()
    
    start_time = time.time()
    try:
        while True:
            # 在实际应用中，这里应该收集实际的系统指标
            # 模拟收集指标
            metrics = monitor.collect_metrics(
                audio_latency=5.0,  # 模拟5ms延迟
                buffer_overflow_count=0,
                frames_processed=100,
                errors_count=0,
                active_streams=1,
                audio_input_device_status="OK"
            )
            
            if args.format == 'json':
                print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))
            else:  # table format
                print(f"Time: {time.ctime(metrics.timestamp)}")
                print(f"CPU Usage: {metrics.cpu_usage}%")
                print(f"Memory Usage: {metrics.memory_usage}%")
                print(f"Audio Latency: {metrics.audio_latency}ms")
                print(f"Buffer Overflow Count: {metrics.buffer_overflow_count}")
                print(f"Frames Processed: {metrics.frames_processed}")
                print(f"Error Count: {metrics.errors_count}")
                print(f"Active Streams: {metrics.active_streams}")
                print("-" * 50)
            
            # 检查是否达到持续时间限制
            if args.duration and (time.time() - start_time) >= args.duration:
                break
                
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


def handle_stop_command(args):
    """Handle stop command"""
    global current_service
    if current_service:
        current_service.stop()
        print("Service stopped")
        current_service = None
    else:
        print("No running service")


if __name__ == "__main__":
    main()
