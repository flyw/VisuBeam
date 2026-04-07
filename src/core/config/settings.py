#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置参数模型
支持 sample_rate, buffer_size, input_channels, mic_positions 等参数
"""
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from .mic_array import MicArrayConfig


from .wpe_config import WpeConfig
# from src.doa.config.doa_config import DOAConfig # Moved to from_dict to avoid circular import
from .aec_config import AecConfig
# from src.enhancement.config.enhancement_config import EnhancementConfig # Moved inside from_dict to avoid circular import


@dataclass
class SystemConfiguration:
    """系统配置参数类"""
    sample_rate: int = 16000  # 采样率，默认16kHz
    buffer_size: int = 1024   # 缓冲区大小，默认1024样本
    shared_buffer_duration_ms: int = 1000  # 共享缓冲区时长（毫秒）
    device_index: Optional[int] = None  # 设备索引
    device_name: Optional[str] = None  # 设备名称
    input_channels: Optional[int] = None  # 输入通道数 (将自动从mic_positions计算，如果不显式指定)
    recording_channels: Optional[List[int]] = None  # 要保存到FLAC的通道索引列表
    mic_positions: List[Dict[str, Any]] = None  # 麦克风位置
    output_port: int = 5000  # 输出端口
    network_protocol: str = "websocket"  # 网络协议
    wpe: Optional[WpeConfig] = None
    doa: Optional[Any] = None # Avoid circular import type check
    # enhancement: Optional['EnhancementConfig'] = None # Forward reference
    enhancement: Optional[Any] = None # Avoid type check issue for now or use string forward ref if TYPE_CHECKING
    aec: Optional[AecConfig] = None

    def __post_init__(self):
        """初始化后验证配置参数"""
        if self.mic_positions is None:
            # 默认4麦克风阵列配置（线性阵列）
            self.mic_positions = [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.05, "y": 0.0, "z": 0.0},
                {"x": 0.1, "y": 0.0, "z": 0.0},
                {"x": 0.15, "y": 0.0, "z": 0.0}
            ]
        
        # 自动计算输入通道数，如果未显式指定
        if self.input_channels is None and self.mic_positions is not None:
            self.input_channels = len(self.mic_positions)
            # print(f"DEBUG: Derived input_channels={self.input_channels} from mic_positions")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfiguration':
        """从字典创建配置对象"""
        from src.enhancement.config.enhancement_config import EnhancementConfig  # Local import to avoid circular dependency
        from src.doa.config.doa_config import DOAConfig # Local import to avoid circular dependency

        audio_config = config_dict.get('audio', {})
        network_config = config_dict.get('network', {})
        wpe_data = config_dict.get('wpe')
        aec_data = config_dict.get('aec')
        doa_data = config_dict.get('doa')
        enhancement_data = config_dict.get('enhancement')


        wpe_config = WpeConfig(wpe_data) if wpe_data else None
        aec_config = AecConfig.from_dict(aec_data) if aec_data else None
        # Pass the full config to DOAConfig as it might need root-level info
        doa_config = DOAConfig(config_dict) if doa_data else None
        enhancement_config = EnhancementConfig(config_dict) if enhancement_data else None

        return cls(
            sample_rate=audio_config.get('sample_rate', 16000),
            buffer_size=audio_config.get('buffer_size', 1024),
            shared_buffer_duration_ms=audio_config.get('shared_buffer_duration_ms', 1000),
            device_index=audio_config.get('device_index'),
            device_name=audio_config.get('device_name'),
            input_channels=audio_config.get('input_channels'),
            recording_channels=audio_config.get('recording_channels'),
            mic_positions=audio_config.get('mic_positions'),
            output_port=network_config.get('output_port', 5000),
            network_protocol=network_config.get('protocol', 'websocket'),
            wpe=wpe_config,
            doa=doa_config,
            enhancement=enhancement_config,
            aec=aec_config
        )
    
    @classmethod
    def from_file(cls, file_path: str) -> 'SystemConfiguration':
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        # This method is not fully compatible with the new structure and should be reviewed if used.
        with open(file_path, 'w', encoding='utf-8') as f:
            # A simple dump for now, might not match original yaml structure perfectly.
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
    
    def validate(self) -> bool:
        """验证配置参数的有效性"""
        # 验证采样率必须为正整数
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive integer, current value: {self.sample_rate}")
        
        # 验证缓冲区大小必须为正整数
        if self.buffer_size <= 0:
            raise ValueError(f"Buffer size must be positive integer, current value: {self.buffer_size}")
        
        # 验证输出端口范围
        if not (1 <= self.output_port <= 65535):
            raise ValueError(f"Output port must be in range 1-65535, current value: {self.output_port}")
        
        # 验证麦克风位置配置
        if self.mic_positions:
            for i, pos in enumerate(self.mic_positions):
                if not all(k in pos for k in ['x', 'y', 'z']):
                    raise ValueError(f"Microphone position {i} missing x, y, z coordinates")
        
        return True

    def update_runtime_parameters(self, **kwargs):
        """
        支持运行时配置参数修改
        
        Args:
            **kwargs: 要更新的参数
        """
        valid_fields = {
            'sample_rate', 'buffer_size', 'device_index', 
            'input_channels', 'mic_positions', 'output_port', 'network_protocol', 
            'enable_wpe', 'save_wpe_output', 'wpe_taps', 'wpe_delay', 'wpe_alpha'
        }
        
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
        
        # 重新验证更新后的配置
        self.validate()


class ConfigurationManager:
    """系统配置管理器，支持运行时配置参数修改"""
    
    def __init__(self, initial_config: SystemConfiguration = None):
        """
        初始化配置管理器
        
        Args:
            initial_config: 初始配置
        """
        self.config = initial_config or SystemConfiguration()
        self.config_history = [self.config.to_dict()]
    
    def update_config(self, **kwargs) -> SystemConfiguration:
        """
        更新配置参数
        
        Args:
            **kwargs: 要更新的配置参数
            
        Returns:
            SystemConfiguration: 更新后的配置
        """
        # 暂存当前配置
        old_config = self.config.to_dict()
        
        try:
            # 尝试更新配置
            self.config.update_runtime_parameters(**kwargs)
            
            # 验证新配置
            self.config.validate()
            
            # 添加到历史记录
            self.config_history.append(self.config.to_dict())
            
            return self.config
            
        except Exception as e:
            # 如果更新失败，恢复原始配置
            self.config = SystemConfiguration.from_dict(old_config)
            raise e
    
    def get_config(self) -> SystemConfiguration:
        """
        获取当前配置
        
        Returns:
            SystemConfiguration: 当前系统配置
        """
        return self.config
    
    def get_config_history(self) -> List[Dict]:
        """
        获取配置历史
        
        Returns:
            List[Dict]: 配置历史列表
        """
        return self.config_history[:]
    
    def reset_to_previous(self) -> SystemConfiguration:
        """
        恢复到上一个配置
        
        Returns:
            SystemConfiguration: 恢复后的配置
        """
        if len(self.config_history) > 1:
            # 移除最新配置，恢复到上一个
            self.config_history.pop()
            prev_config = self.config_history[-1]
            self.config = SystemConfiguration.from_dict(prev_config)
        
        return self.config
    
    def save_config_to_file(self, filepath: str):
        """
        保存配置到文件
        
        Args:
            filepath: 文件路径
        """
        self.config.save_to_file(filepath)
    
    def load_config_from_file(self, filepath: str) -> SystemConfiguration:
        """
        从文件加载配置
        
        Args:
            filepath: 文件路径
            
        Returns:
            SystemConfiguration: 加载的配置
        """
        self.config = SystemConfiguration.from_file(filepath)
        self.config.validate()
        self.config_history.append(self.config.to_dict())
        return self.config
