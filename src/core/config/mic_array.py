#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
麦克风阵列配置类
支持麦克风位置配置
"""
from typing import Dict, List, Optional
import math


class MicArrayConfig:
    """麦克风阵列配置类"""
    
    def __init__(self, positions: List[Dict[str, float]] = None):
        """
        初始化麦克风阵列配置
        
        Args:
            positions: 麦克风位置列表，每个位置包含x, y, z坐标
        """
        if positions is None:
            # 默认4麦克风线性阵列
            self.positions = [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.05, "y": 0.0, "z": 0.0},  # 5cm间距
                {"x": 0.1, "y": 0.0, "z": 0.0},   # 10cm间距
                {"x": 0.15, "y": 0.0, "z": 0.0}   # 15cm间距
            ]
        else:
            self.positions = positions
        
        self.validate()
    
    def validate(self):
        """验证麦克风位置配置的有效性"""
        if not self.positions:
            raise ValueError("Microphone position configuration cannot be empty")

        for i, pos in enumerate(self.positions):
            if not all(k in pos for k in ['x', 'y', 'z']):
                raise ValueError(f"Microphone position {i} missing x, y, z coordinates")

    def get_mic_count(self) -> int:
        """获取麦克风数量"""
        return len(self.positions)
    
    def get_distance_between_mics(self, mic1_idx: int, mic2_idx: int) -> float:
        """计算两个麦克风之间的距离"""
        if mic1_idx >= len(self.positions) or mic2_idx >= len(self.positions):
            raise IndexError("Microphone index out of range")
        
        pos1 = self.positions[mic1_idx]
        pos2 = self.positions[mic2_idx]
        
        dx = pos2['x'] - pos1['x']
        dy = pos2['y'] - pos1['y']
        dz = pos2['z'] - pos1['z']
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_positions(self) -> List[Dict[str, float]]:
        """获取麦克风位置列表"""
        return self.positions[:]
    
    def to_dict(self) -> List[Dict[str, float]]:
        """转换为字典格式"""
        return self.positions[:]
    
    @classmethod
    def from_dict(cls, positions_dict: List[Dict[str, float]]) -> 'MicArrayConfig':
        """从字典创建麦克风阵列配置"""
        return cls(positions_dict)
