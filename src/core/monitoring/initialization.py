#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统初始化监控模块
实现T048: 实现系统初始化成功率监控 (SC-001)
"""
import time
from typing import Dict, List
from datetime import datetime


class InitializationMonitor:
    """系统初始化监控器"""
    
    def __init__(self):
        self.initialization_attempts = 0
        self.successful_initializations = 0
        self.failed_initializations = 0
        self.start_time = time.time()
        self.initialization_records: List[Dict] = []  # 记录初始化尝试
    
    def record_initialization_attempt(self, success: bool, message: str = ""):
        """
        记录初始化尝试
        
        Args:
            success: 初始化是否成功
            message: 初始化消息或错误信息
        """
        self.initialization_attempts += 1
        timestamp = time.time()
        
        record = {
            'timestamp': timestamp,
            'success': success,
            'message': message,
            'attempt_number': self.initialization_attempts
        }
        
        self.initialization_records.append(record)
        
        if success:
            self.successful_initializations += 1
        else:
            self.failed_initializations += 1
    
    def get_initialization_success_rate(self) -> float:
        """
        获取初始化成功率
        
        Returns:
            float: 初始化成功率 (0.0 - 1.0)
        """
        if self.initialization_attempts == 0:
            return 1.0  # 没有尝试时，认为是100%成功
        
        return self.successful_initializations / self.initialization_attempts
    
    def get_success_percentage(self) -> float:
        """
        获取初始化成功率百分比
        
        Returns:
            float: 初始化成功率百分比
        """
        return self.get_initialization_success_rate() * 100.0
    
    def is_meeting_success_criteria(self) -> bool:
        """
        检查是否满足成功率标准 (SC-001: 99%以上)
        
        Returns:
            bool: 是否满足成功率标准
        """
        return self.get_initialization_success_rate() >= 0.99
    
    def get_statistics(self) -> Dict:
        """
        获取初始化统计信息
        
        Returns:
            Dict: 统计信息
        """
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_attempts': self.initialization_attempts,
            'successful_initializations': self.successful_initializations,
            'failed_initializations': self.failed_initializations,
            'success_rate': self.get_initialization_success_rate(),
            'success_percentage': self.get_success_percentage(),
            'meeting_criteria': self.is_meeting_success_criteria(),
            'uptime_seconds': elapsed_time,
            'records_count': len(self.initialization_records)
        }
    
    def get_recent_records(self, count: int = 10) -> List[Dict]:
        """
        获取最近的初始化记录
        
        Args:
            count: 要获取的记录数量
            
        Returns:
            List[Dict]: 最近的初始化记录
        """
        return self.initialization_records[-count:] if self.initialization_records else []
    
    def reset_statistics(self):
        """重置统计信息"""
        self.initialization_attempts = 0
        self.successful_initializations = 0
        self.failed_initializations = 0
        self.start_time = time.time()
        self.initialization_records = []


def create_initialization_monitor() -> InitializationMonitor:
    """
    创建初始化监控器
    
    Returns:
        InitializationMonitor: 初始化监控器
    """
    return InitializationMonitor()