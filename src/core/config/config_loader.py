#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载方法
支持从JSON文件加载配置
"""
import json
import os
from typing import Dict, Any, Optional
from .settings import SystemConfiguration


def load_config_from_file(config_path: str) -> SystemConfiguration:
    """
    Load system configuration from JSON or YAML file
    
    Args:
        config_path: Configuration file path
        
    Returns:
        SystemConfiguration: System configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    # Check file extension to determine format
    from pathlib import Path
    file_path = Path(config_path)
    file_ext = file_path.suffix.lower()
    
    # Try importing yaml module
    try:
        import yaml
        HAS_YAML = True
    except ImportError:
        HAS_YAML = False
    
    if file_ext in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML configuration files. Install with: pip install PyYAML")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    else:  # assume JSON
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    
    return SystemConfiguration.from_dict(config_dict)

def save_config_to_file(config: SystemConfiguration, config_path: str):
    """
    将系统配置保存到JSON文件
    
    Args:
        config: 系统配置对象
        config_path: 配置文件路径
    """
    config.save_to_file(config_path)


def get_default_config() -> SystemConfiguration:
    """
    获取默认系统配置
    
    Returns:
        SystemConfiguration: 默认系统配置对象
    """
    return SystemConfiguration()


def validate_config(config: SystemConfiguration) -> bool:
    """
    验证配置参数的有效性
    
    Args:
        config: 系统配置对象
        
    Returns:
        bool: 配置是否有效
    """
    return config.validate()


def load_doa_config_from_file(config_path: str):
    """
    Load complete DOA configuration from YAML file, including all settings

    Args:
        config_path: Configuration file path

    Returns:
        DOAConfig: DOA configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    # Check file extension to determine format
    from pathlib import Path
    file_path = Path(config_path)
    file_ext = file_path.suffix.lower()

    # Try importing yaml module
    try:
        import yaml
        HAS_YAML = True
    except ImportError:
        HAS_YAML = False

    if file_ext in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML configuration files. Install with: pip install PyYAML")

        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
    else:  # assume JSON
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)

    # Create a new dict that contains complete configuration with necessary audio settings
    from src.doa.config.doa_config import DOAConfig

    # Start with the original full config to preserve all sections
    doa_config_data = {}

    # Copy all sections from the original config
    for key, value in full_config.items():
        doa_config_data[key] = value if not isinstance(value, dict) else value.copy()

    # Ensure we have audio settings available in the config at the top level
    audio_config = full_config.get("audio", {})

    # Override audio settings in the DOA config with those from audio section
    doa_config_data["sample_rate"] = audio_config.get("sample_rate", 16000)
    input_channels = audio_config.get("input_channels")
    if input_channels is None:
        # If input_channels not set, derive from mic_positions count
        input_channels = len(audio_config.get("mic_positions", []))

    doa_config_data["num_mics"] = input_channels if input_channels else 4
    doa_config_data["mic_positions"] = audio_config.get("mic_positions", [])
    doa_config_data["buffer_size"] = audio_config.get("buffer_size", 1024)

    return DOAConfig(doa_config_data)


def merge_configs(base_config: SystemConfiguration, override_config: SystemConfiguration) -> SystemConfiguration:
    """
    合并两个配置，后一个配置会覆盖前一个配置的相同字段

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        SystemConfiguration: 合并后的配置
    """
    # 如果覆盖配置的字段不为None，则使用覆盖配置的值
    sample_rate = override_config.sample_rate if override_config.sample_rate != 16000 else base_config.sample_rate
    buffer_size = override_config.buffer_size if override_config.buffer_size != 1024 else base_config.buffer_size
    device_index = override_config.device_index if override_config.device_index is not None else base_config.device_index
    input_channels = override_config.input_channels if override_config.input_channels is not None else base_config.input_channels
    mic_positions = override_config.mic_positions if override_config.mic_positions is not None else base_config.mic_positions
    output_port = override_config.output_port if override_config.output_port != 5000 else base_config.output_port
    network_protocol = override_config.network_protocol if override_config.network_protocol != "websocket" else base_config.network_protocol

    merged_config = SystemConfiguration(
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        device_index=device_index,
        input_channels=input_channels,
        mic_positions=mic_positions,
        output_port=output_port,
        network_protocol=network_protocol
    )

    return merged_config