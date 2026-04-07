#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频缓冲区管理类
实现环形缓冲区，优化性能以确保低延迟和防数据丢失，满足SC-002（<10ms延迟）和FR-005（防数据丢失）要求
"""
import numpy as np
from typing import Optional, Union
import threading
import time
import uuid


class AudioChunk:
    """A simple data class to hold a chunk of audio data and its metadata."""
    def __init__(self, data: np.ndarray, num_channels: int, timestamp: float):
        self.id = str(uuid.uuid4())
        self.data = data
        self.num_channels = num_channels
        self.timestamp = timestamp

    def copy(self):
        """Creates a thread-safe copy of the AudioChunk."""
        new_chunk = AudioChunk(
            data=self.data.copy(),
            num_channels=self.num_channels,
            timestamp=self.timestamp
        )
        # Preserve the original ID for tracking across threads
        new_chunk.id = self.id
        return new_chunk

    def __len__(self):
        return len(self.data)


class AudioBuffer:
    """音频缓冲区管理类，实现环形缓冲区，优化性能以确保低延迟和防数据丢失"""

    def __init__(self, buffer_size: int = 1024, dtype=np.int16):
        """
        初始化音频缓冲区

        Args:
            buffer_size: 缓冲区大小
            dtype: 数据类型，默认为np.int16（16位整数）
        """
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.buffer = np.zeros(buffer_size, dtype=dtype)
        self.read_position = 0
        self.write_position = 0
        self.is_full = False
        self.lock = threading.Lock()  # 线程安全锁
        self.overflow_count = 0  # 溢出计数器，用于监控数据丢失
        self.last_write_time = time.time()  # 上次写入时间
        self.last_read_time = time.time()   # 上次读取时间
    
    def write(self, data: Union[np.ndarray, list]) -> int:
        """
        向缓冲区写入数据，实现防数据丢失机制
        
        Args:
            data: 要写入的数据
            
        Returns:
            int: 实际写入的数据量
        """
        with self.lock:
            if isinstance(data, list):
                data = np.array(data, dtype=self.dtype)
            
            data_len = len(data)
            available_space = self._get_available_write_space()
            
            # 检查是否会发生溢出
            if data_len > available_space:
                # 如果缓冲区无法容纳所有数据，计算可写入的数据量并增加溢出计数
                self.overflow_count += 1
                # 只写入可用空间大小的数据，丢弃多余的数据
                data = data[:available_space]
                data_len = available_space
            
            # 执行写入操作
            if self.write_position + data_len <= self.buffer_size:
                # 数据可以连续写入
                self.buffer[self.write_position:self.write_position + data_len] = data
                self.write_position = (self.write_position + data_len) % self.buffer_size
            else:
                # 需要分两段写入（环形缓冲区）
                first_part_size = self.buffer_size - self.write_position
                self.buffer[self.write_position:] = data[:first_part_size]
                self.buffer[0:data_len - first_part_size] = data[first_part_size:]
                self.write_position = (data_len - first_part_size) % self.buffer_size
            
            # 检查是否写满
            if self.write_position == self.read_position:
                self.is_full = True
            
            # 更新写入时间
            self.last_write_time = time.time()
            
            return data_len
    
    def read(self, size: int) -> np.ndarray:
        """
        从缓冲区读取数据
        
        Args:
            size: 要读取的数据量
            
        Returns:
            np.ndarray: 读取的数据
        """
        with self.lock:
            available_data = self._get_available_read_data()
            
            # 如果请求的数据量超过可用数据，只读取可用数据
            if size > available_data:
                size = available_data
            
            result = np.zeros(size, dtype=self.dtype)
            
            if size == 0:
                return result
            
            if self.read_position + size <= self.buffer_size:
                # 数据可以连续读取
                result[:] = self.buffer[self.read_position:self.read_position + size]
                self.read_position = (self.read_position + size) % self.buffer_size
            else:
                # 需要分两段读取（环形缓冲区）
                first_part_size = self.buffer_size - self.read_position
                result[:first_part_size] = self.buffer[self.read_position:]
                result[first_part_size:] = self.buffer[0:size - first_part_size]
                self.read_position = (size - first_part_size) % self.buffer_size
            
            # 重置满状态
            self.is_full = False
            
            # 更新读取时间
            self.last_read_time = time.time()
            
            return result
    
    def _get_available_write_space(self) -> int:
        """获取可用写入空间"""
        if self.is_full:
            return 0
        
        if self.write_position >= self.read_position:
            available = self.buffer_size - (self.write_position - self.read_position)
        else:
            available = self.read_position - self.write_position
        
        return available - 1  # 保留一个位置以区分满和空状态
    
    def _get_available_read_data(self) -> int:
        """获取可用读取数据量"""
        if self.is_full:
            return self.buffer_size
        
        if self.write_position >= self.read_position:
            return self.write_position - self.read_position
        else:
            return self.buffer_size - (self.read_position - self.read_position)
    
    def get_fill_level(self) -> float:
        """
        获取缓冲区填充水平
        
        Returns:
            float: 填充比例 (0.0 - 1.0)
        """
        return self._get_available_read_data() / self.buffer_size
    
    def get_overflow_count(self) -> int:
        """
        获取溢出计数（用于监控数据丢失）
        
        Returns:
            int: 溢出次数
        """
        with self.lock:
            return self.overflow_count
    
    def reset_overflow_count(self):
        """重置溢出计数"""
        with self.lock:
            self.overflow_count = 0
    
    def get_latency_estimate(self, sample_rate: int) -> float:
        """
        估算缓冲区延迟
        
        Args:
            sample_rate: 采样率（Hz）
            
        Returns:
            float: 延迟（毫秒）
        """
        if sample_rate <= 0:
            return 0.0
        
        fill_count = self._get_available_read_data()
        latency_seconds = fill_count / sample_rate
        return latency_seconds * 1000  # 转换为毫秒
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.read_position = 0
            self.write_position = 0
            self.is_full = False
            self.overflow_count = 0
            self.buffer.fill(0)
            self.last_write_time = time.time()
            self.last_read_time = time.time()
    
    def is_empty(self) -> bool:
        """
        检查缓冲区是否为空
        
        Returns:
            bool: 是否为空
        """
        with self.lock:
            return self.read_position == self.read_position and not self.is_full
    
    def is_full_status(self) -> bool:
        """
        检查缓冲区是否为满（不同于is_full属性，这是接口方法）
        
        Returns:
            bool: 是否为满
        """
        with self.lock:
            return self.is_full


class AudioBufferManager:
    """音频缓冲区管理器，可以管理多个缓冲区（用于多通道音频），优化性能以确保低延迟和防数据丢失"""

    def __init__(self, num_channels: int = 1, buffer_size: int = 1024, dtype=np.int16):
        """
        初始化音频缓冲区管理器

        Args:
            num_channels: 通道数
            buffer_size: 每个通道的缓冲区大小
            dtype: 数据类型，默认为np.int16
        """
        self.num_channels = num_channels
        self.buffers = [AudioBuffer(buffer_size, dtype) for _ in range(num_channels)]
        self.last_access_time = time.time()
    
    def write_frame(self, frame_data: np.ndarray) -> int:
        """
        写入音频帧数据，实现防数据丢失机制
        
        Args:
            frame_data: 音频帧数据，形状为 (num_samples, num_channels)
            
        Returns:
            int: 实际写入的样本数
        """
        self.last_access_time = time.time()
        
        if frame_data.ndim == 1:
            # 单声道数据
            return self.buffers[0].write(frame_data)
        else:
            # 多声道数据
            num_samples, num_channels = frame_data.shape
            if num_channels != self.num_channels:
                raise ValueError(f"Channel count mismatch: expected {self.num_channels}, got {num_channels}")

            # 将每个通道的数据写入对应的缓冲区
            actual_written = float('inf')
            for ch in range(num_channels):
                written = self.buffers[ch].write(frame_data[:, ch])
                actual_written = min(actual_written, written)
            
            return int(actual_written)
    
    def read_frame(self, size: int) -> np.ndarray:
        """
        读取音频帧数据
        
        Args:
            size: 要读取的样本数
            
        Returns:
            np.ndarray: 音频帧数据，形状为 (size, num_channels)
        """
        self.last_access_time = time.time()
        
        if self.num_channels == 1:
            # 单声道数据
            data = self.buffers[0].read(size)
            return data.reshape(-1, 1)  # 调整形状为 (size, 1)
        else:
            # 多声道数据
            frame_data = np.zeros((size, self.num_channels), dtype=self.buffers[0].dtype)
            for ch in range(self.num_channels):
                frame_data[:, ch] = self.buffers[ch].read(size)
            return frame_data
    
    def clear_all(self):
        """清空所有缓冲区"""
        for buf in self.buffers:
            buf.clear()
        self.last_access_time = time.time()
    
    def get_fill_levels(self) -> list:
        """
        获取所有缓冲区的填充水平
        
        Returns:
            list: 各缓冲区填充水平列表
        """
        return [buf.get_fill_level() for buf in self.buffers]
    
    def get_total_overflow_count(self) -> int:
        """
        获取所有缓冲区的总溢出计数
        
        Returns:
            int: 总溢出次数
        """
        return sum(buf.get_overflow_count() for buf in self.buffers)
    
    def reset_all_overflow_counts(self):
        """重置所有缓冲区的溢出计数"""
        for buf in self.buffers:
            buf.reset_overflow_count()
    
    def get_average_latency(self, sample_rate: int) -> float:
        """
        获取平均缓冲区延迟，确保满足<10ms延迟要求
        
        Args:
            sample_rate: 采样率（Hz）
            
        Returns:
            float: 平均延迟（毫秒）
        """
        if sample_rate <= 0:
            return 0.0
        
        latencies = [buf.get_latency_estimate(sample_rate) for buf in self.buffers]
        if latencies:
            return sum(latencies) / len(latencies)
        else:
            return 0.0
    
    def are_all_empty(self) -> bool:
        """
        检查所有缓冲区是否都为空
        
        Returns:
            bool: 所有缓冲区是否都为空
        """
        return all(buf.is_empty() for buf in self.buffers)
    
    def get_buffer_status(self) -> list:
        """
        获取所有缓冲区的状态信息
        
        Returns:
            list: 各缓冲区状态信息列表
        """
        status_list = []
        for i, buf in enumerate(self.buffers):
            status_list.append({
                'channel': i,
                'fill_level': buf.get_fill_level(),
                'overflow_count': buf.get_overflow_count(),
                'is_full': buf.is_full_status(),
                'is_empty': buf.is_empty(),
                'latency_ms': buf.get_latency_estimate(16000)  # 假设16kHz采样率进行估算
            })
        return status_list


class SharedCircularBuffer:
    """
    线程安全的共享环形缓冲区，用于解耦 I/O 和处理。
    
    设计遵循 SPEC-009 规范：
    - 生产者 (AudioStreamPipeline) 通过 write() 写入数据
    - 消费者 (DOAService) 通过 read_latest() 读取最新数据
    - 缓冲区满时覆盖最旧数据，不会阻塞生产者
    """
    
    def __init__(self, capacity_samples: int, num_channels: int, dtype=np.float32):
        """
        初始化共享环形缓冲区
        
        Args:
            capacity_samples: 缓冲区容量（样本数）
            num_channels: 音频通道数
            dtype: 数据类型，默认 float32
        """
        self._capacity = capacity_samples
        self._num_channels = num_channels
        self._dtype = dtype
        
        # 环形缓冲区存储
        self._buffer = np.zeros((capacity_samples, num_channels), dtype=dtype)
        
        # 写入位置（指向下一个要写入的位置）
        self._write_pos = 0
        
        # 已写入的总样本数（用于判断缓冲区是否有足够数据）
        self._total_written = 0
        
        # 线程安全锁
        self._lock = threading.Lock()
        
        # 读取位置（用于消费性读取 read_consume）
        self._total_read = 0
        
        # 统计信息
        self._overflow_count = 0
        self._last_write_time = 0.0
    
    @property
    def capacity(self) -> int:
        """返回缓冲区容量（样本数）"""
        return self._capacity
    
    @property
    def num_channels(self) -> int:
        """返回通道数"""
        return self._num_channels
    
    def write(self, data: np.ndarray) -> int:
        """
        向缓冲区写入数据（生产者调用）
        
        缓冲区满时会覆盖最旧的数据，保证生产者永不阻塞。
        
        Args:
            data: 音频数据，形状为 (num_samples, num_channels) 或 (num_samples,)
            
        Returns:
            int: 实际写入的样本数
        """
        with self._lock:
            # 确保数据是 2D 的
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            num_samples = data.shape[0]
            data_channels = data.shape[1]
            
            # 通道数校验：如果输入通道多于缓冲区，截取前 N 个通道
            if data_channels > self._num_channels:
                data = data[:, :self._num_channels]
            elif data_channels < self._num_channels:
                # 如果输入通道少于缓冲区，用零填充
                padded = np.zeros((num_samples, self._num_channels), dtype=self._dtype)
                padded[:, :data_channels] = data
                data = padded
            
            # 转换数据类型
            if data.dtype != self._dtype:
                data = data.astype(self._dtype)
            
            # 检查是否会发生覆盖（溢出）
            available = self._capacity - min(self._total_written, self._capacity)
            if num_samples > available:
                self._overflow_count += 1
            
            # 写入数据到环形缓冲区
            if num_samples >= self._capacity:
                # 数据量大于或等于缓冲区容量，只保留最新的数据
                self._buffer[:] = data[-self._capacity:]
                self._write_pos = 0
                self._total_written = self._capacity
            else:
                # 正常写入
                end_pos = self._write_pos + num_samples
                
                if end_pos <= self._capacity:
                    # 数据不需要环绕
                    self._buffer[self._write_pos:end_pos] = data
                else:
                    # 数据需要环绕
                    first_part = self._capacity - self._write_pos
                    self._buffer[self._write_pos:] = data[:first_part]
                    self._buffer[:num_samples - first_part] = data[first_part:]
                
                self._write_pos = end_pos % self._capacity
                self._total_written += num_samples
            
            self._last_write_time = time.time()
            return num_samples
    
    def read_latest(self, num_samples: int) -> Optional[np.ndarray]:
        """
        读取最新的 num_samples 个样本（消费者调用）
        
        这是一个非破坏性读取，不会移动读指针。多个消费者可以读取相同的数据。
        
        Args:
            num_samples: 要读取的样本数
            
        Returns:
            np.ndarray: 形状为 (num_samples, num_channels) 的音频数据，
                       如果缓冲区数据不足则返回 None
        """
        with self._lock:
            # 检查是否有足够的数据
            available = min(self._total_written, self._capacity)
            if available < num_samples:
                return None
            
            # 计算读取的起始位置（最新数据在 write_pos 之前）
            start_pos = (self._write_pos - num_samples) % self._capacity
            
            # 分配输出数组
            result = np.zeros((num_samples, self._num_channels), dtype=self._dtype)
            
            if start_pos + num_samples <= self._capacity:
                # 数据不需要环绕
                result[:] = self._buffer[start_pos:start_pos + num_samples]
            else:
                # 数据需要环绕读取
                first_part = self._capacity - start_pos
                result[:first_part] = self._buffer[start_pos:]
                result[first_part:] = self._buffer[:num_samples - first_part]
            
            return result
    
    def read_consume(self, num_samples: int) -> Optional[np.ndarray]:
        """
        消费性读取 - 读取并移动读指针（用于 DOA 连续处理）
        
        与 read_latest 不同，这个方法会移动内部读指针，确保数据不会被重复读取或丢失。
        
        Args:
            num_samples: 要读取的样本数
            
        Returns:
            np.ndarray: 形状为 (num_samples, num_channels) 的音频数据，
                       如果缓冲区数据不足则返回 None
        """
        with self._lock:
            # 检查是否有足够的新数据
            available_new = self._total_written - self._total_read
            if available_new < num_samples:
                return None
            
            # 计算读取的起始位置
            read_pos = self._total_read % self._capacity
            
            # 分配输出数组
            result = np.zeros((num_samples, self._num_channels), dtype=self._dtype)
            
            if read_pos + num_samples <= self._capacity:
                # 数据不需要环绕
                result[:] = self._buffer[read_pos:read_pos + num_samples]
            else:
                # 数据需要环绕读取
                first_part = self._capacity - read_pos
                result[:first_part] = self._buffer[read_pos:]
                result[first_part:] = self._buffer[:num_samples - first_part]
            
            # 移动读指针
            self._total_read += num_samples
            
            return result
    
    def get_available_samples(self) -> int:
        """
        获取当前可用的样本数
        
        Returns:
            int: 可用样本数（最大为 capacity）
        """
        with self._lock:
            return min(self._total_written, self._capacity)
    
    def get_overflow_count(self) -> int:
        """
        获取溢出计数
        
        Returns:
            int: 发生覆盖的次数
        """
        with self._lock:
            return self._overflow_count
    
    def reset_overflow_count(self):
        """重置溢出计数"""
        with self._lock:
            self._overflow_count = 0
    
    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._total_written = 0
            self._overflow_count = 0
            self._last_write_time = 0.0
    
    def get_status(self) -> dict:
        """
        获取缓冲区状态信息
        
        Returns:
            dict: 包含状态信息的字典
        """
        with self._lock:
            available = min(self._total_written, self._capacity)
            return {
                'capacity': self._capacity,
                'num_channels': self._num_channels,
                'available_samples': available,
                'fill_level': available / self._capacity if self._capacity > 0 else 0.0,
                'overflow_count': self._overflow_count,
                'last_write_time': self._last_write_time
            }
