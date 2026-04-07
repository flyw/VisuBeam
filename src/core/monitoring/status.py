#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控数据类
创建系统监控数据类 src/core/monitoring/status.py
需求实体: 系统监控数据，包含CPU使用率、内存使用情况、音频延迟等系统运行指标
"""
import time
import psutil
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SystemMonitoringData:
    """实现CPU使用率、内存使用情况、音频延迟等监控指标 (T044)"""
    
    timestamp: float
    cpu_usage: float  # CPU使用率 (%)
    memory_usage: float  # 内存使用率 (%)
    audio_latency: float  # 音频延迟 (毫秒)
    buffer_overflow_count: int  # 缓冲区溢出次数
    frames_processed: int  # 已处理帧数
    errors_count: int  # 错误次数
    active_streams: int  # 活跃音频流数量
    audio_input_device_status: str = "UNKNOWN"  # 音频输入设备状态
    system_uptime: float = 0.0  # 系统运行时间（秒）
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemMonitoringData':
        """从字典创建监控数据对象"""
        return cls(**data)
    
    @classmethod
    def collect_current_metrics(cls, 
                              audio_latency: float = 0.0,
                              buffer_overflow_count: int = 0,
                              frames_processed: int = 0,
                              errors_count: int = 0,
                              active_streams: int = 0,
                              audio_input_device_status: str = "UNKNOWN",
                              system_uptime: float = 0.0) -> 'SystemMonitoringData':
        """
        收集当前系统指标 (CPU使用率、内存使用情况、音频延迟等)
        
        Args:
            audio_latency: 音频延迟 (毫秒)
            buffer_overflow_count: 缓冲区溢出次数
            frames_processed: 已处理帧数
            errors_count: 错误次数
            active_streams: 活跃音频流数量
            audio_input_device_status: 音频输入设备状态
            system_uptime: 系统运行时间（秒）
            
        Returns:
            SystemMonitoringData: 当前系统监控数据
        """
        # 收集系统级指标
        timestamp = time.time()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        return cls(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            audio_latency=audio_latency,
            buffer_overflow_count=buffer_overflow_count,
            frames_processed=frames_processed,
            errors_count=errors_count,
            active_streams=active_streams,
            audio_input_device_status=audio_input_device_status,
            system_uptime=system_uptime
        )


class SystemMonitor:
    """实现系统运行状态监控功能 (FR-006) - 系统操作监控"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history: List[SystemMonitoringData] = []
        self.total_frames_processed = 0
        self.total_errors = 0
        self.active_streams_count = 0
        self.buffer_overflow_total = 0
        
    def collect_metrics(self, 
                       audio_latency: float = 0.0,
                       buffer_overflow_count: int = 0,
                       frames_processed: int = 0,
                       errors_count: int = 0,
                       active_streams: int = 0,
                       audio_input_device_status: str = "UNKNOWN") -> SystemMonitoringData:
        """
        收集系统指标
        
        Args:
            audio_latency: 音频延迟 (毫秒)
            buffer_overflow_count: 本次收集的缓冲区溢出次数
            frames_processed: 本次收集的已处理帧数
            errors_count: 本次收集的错误次数
            active_streams: 活跃音频流数量
            audio_input_device_status: 音频输入设备状态
            
        Returns:
            SystemMonitoringData: 收集到的监控数据
        """
        # 累计计数
        self.total_frames_processed += frames_processed
        self.total_errors += errors_count
        self.active_streams_count = active_streams
        self.buffer_overflow_total += buffer_overflow_count
        
        # 计算系统运行时间
        system_uptime = time.time() - self.start_time
        
        # 收集当前指标
        current_metrics = SystemMonitoringData.collect_current_metrics(
            audio_latency=audio_latency,
            buffer_overflow_count=self.buffer_overflow_total,
            frames_processed=self.total_frames_processed,
            errors_count=self.total_errors,
            active_streams=active_streams,
            audio_input_device_status=audio_input_device_status,
            system_uptime=system_uptime
        )
        
        # 添加到历史记录
        self.metrics_history.append(current_metrics)
        
        # 限制历史记录大小，避免内存无限增长
        if len(self.metrics_history) > 1000:  # 保留最近1000条记录
            self.metrics_history = self.metrics_history[-500:]  # 保留后500条
        
        return current_metrics
    
    def get_current_metrics(self) -> SystemMonitoringData:
        """
        获取当前监控指标
        
        Returns:
            SystemMonitoringData: 最新的监控数据
        """
        if not self.metrics_history:
            # 如果没有历史数据，返回当前指标
            system_uptime = time.time() - self.start_time
            return SystemMonitoringData.collect_current_metrics(
                system_uptime=system_uptime
            )
        
        return self.metrics_history[-1]
    
    def get_average_metrics(self, duration_seconds: int = 60) -> Optional[SystemMonitoringData]:
        """
        获取指定时间内的平均监控指标
        
        Args:
            duration_seconds: 统计时间窗口（秒）
            
        Returns:
            SystemMonitoringData: 平均监控数据，如果没有足够数据则返回None
        """
        if not self.metrics_history:
            return None
        
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history 
            if current_time - m.timestamp <= duration_seconds
        ]
        
        if not recent_metrics:
            return None
        
        # 计算平均值
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_audio_latency = sum(m.audio_latency for m in recent_metrics) / len(recent_metrics)
        
        # 使用最新的计数信息
        latest = recent_metrics[-1]
        
        return SystemMonitoringData(
            timestamp=latest.timestamp,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            audio_latency=avg_audio_latency,
            buffer_overflow_count=latest.buffer_overflow_count,
            frames_processed=latest.frames_processed,
            errors_count=latest.errors_count,
            active_streams=latest.active_streams,
            audio_input_device_status=latest.audio_input_device_status,
            system_uptime=latest.system_uptime
        )
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[SystemMonitoringData]:
        """
        获取指定时间内的监控指标历史
        
        Args:
            duration_seconds: 时间窗口（秒）
            
        Returns:
            List[SystemMonitoringData]: 监控数据历史列表
        """
        current_time = time.time()
        return [
            m for m in self.metrics_history 
            if current_time - m.timestamp <= duration_seconds
        ]
    
    def reset_counters(self):
        """重置计数器"""
        self.total_frames_processed = 0
        self.total_errors = 0
        self.buffer_overflow_total = 0
        self.metrics_history.clear()
        self.start_time = time.time()
    
    def get_system_status_summary(self) -> Dict:
        """
        获取系统状态摘要
        
        Returns:
            Dict: 系统状态摘要
        """
        current_metrics = self.get_current_metrics()
        
        return {
            'timestamp': current_metrics.timestamp,
            'cpu_usage': current_metrics.cpu_usage,
            'memory_usage': current_metrics.memory_usage,
            'audio_latency': current_metrics.audio_latency,
            'buffer_overflow_count': current_metrics.buffer_overflow_count,
            'frames_processed': current_metrics.frames_processed,
            'errors_count': current_metrics.errors_count,
            'active_streams': current_metrics.active_streams,
            'audio_input_device_status': current_metrics.audio_input_device_status,
            'system_uptime': current_metrics.system_uptime,
            'status': self._determine_status(current_metrics)
        }
    
    def _determine_status(self, metrics: SystemMonitoringData) -> str:
        """
        根据监控数据确定系统状态
        
        Args:
            metrics: 监控数据
            
        Returns:
            str: 系统状态 ('OK', 'WARNING', 'ERROR')
        """
        # 检查各种问题指标
        if metrics.audio_input_device_status == "ERROR" or metrics.errors_count > 0:
            return "ERROR"
        
        if (metrics.cpu_usage > 80.0 or 
            metrics.memory_usage > 80.0 or 
            metrics.audio_latency > 10.0 or  # 超过10ms延迟
            metrics.buffer_overflow_count > 0):
            return "WARNING"
        
        return "OK"