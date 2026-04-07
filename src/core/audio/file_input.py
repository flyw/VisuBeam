#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频文件输入处理器
支持WAV、FLAC等格式
实现T027: [P] [US2] 创建音频文件输入处理器 src/core/audio/file_input.py，支持WAV、FLAC等格式
"""
import numpy as np
import os
from typing import Optional, Tuple, Union
from pathlib import Path


class AudioFileProcessor:
    """音频文件处理器，支持加载预录制的多通道音频数据"""

    SUPPORTED_FORMATS = ('.wav', '.flac', '.mp3', '.aac', '.m4a', '.aiff', '.aif')

    def __init__(self, file_path: str):
        """
        初始化音频文件处理器

        Args:
            file_path: 音频文件路径
        """
        self.file_path = Path(file_path)
        self.audio_data = None
        self.sample_rate = None
        self.channel_count = None
        self.duration = None
        self.is_loaded = False
        self.current_position = 0

        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        if self.file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的音频文件格式 {self.file_path.suffix}，支持的格式: {self.SUPPORTED_FORMATS}")

    def load_audio(self):
        """加载音频文件"""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("Need to install soundfile library for audio file processing: pip install soundfile")

        try:
            # 使用soundfile读取音频数据
            self.audio_data, self.sample_rate = sf.read(str(self.file_path))

            # 确定通道数
            if self.audio_data.ndim == 1:
                # 単声道
                self.channel_count = 1
                # 重塑为 (N, 1) 格式以便与其他代码兼容
                self.audio_data = self.audio_data.reshape(-1, 1)
            else:
                # 多声道
                self.channel_count = self.audio_data.shape[1]

            # 计算持续时间（秒）
            self.duration = len(self.audio_data) / self.sample_rate if self.sample_rate > 0 else 0.0

            self.is_loaded = True
            self.current_position = 0

        except Exception as e:
            raise Exception(f"Failed to load audio file: {str(e)}")

    def load_audio_with_conversion(self, target_sample_rate: int = None, target_channels: int = None):
        """
        加载音频文件并根据需要转换参数
        处理边界情况：当音频文件的采样率或通道数与配置不匹配时的行为 (T061)
        
        Args:
            target_sample_rate: 目标采样率
            target_channels: 目标通道数
        """
        self.load_audio()

        needs_conversion = False

        # 检查是否需要转换
        if target_sample_rate and self.sample_rate != target_sample_rate:
            needs_conversion = True
            print(
                f"Warning: Audio file sample rate {self.sample_rate}Hz does not match target {target_sample_rate}Hz, "
                f"requires conversion")

        if target_channels and self.channel_count != target_channels:
            needs_conversion = True
            print(
                f"Warning: Audio file channel count {self.channel_count} does not match target {target_channels}, "
                f"requires conversion")

        if needs_conversion:
            # 执行转换
            try:
                import resampy
                original_data = self.audio_data.copy()
                original_rate = self.sample_rate

                # 先转换采样率
                if target_sample_rate and self.sample_rate != target_sample_rate:
                    if self.audio_data.shape[1] == 1:
                        # 単声道
                        converted_data = resampy.resample(original_data.flatten(), original_rate, target_sample_rate)
                        self.audio_data = converted_data.reshape(-1, 1)
                    else:
                        # 多声道，逐个声道重采样
                        resampled_channels = []
                        for ch in range(original_data.shape[1]):
                            ch_data = resampy.resample(original_data[:, ch], original_rate, target_sample_rate)
                            resampled_channels.append(ch_data)
                        # 重新组合多声道数据
                        self.audio_data = np.column_stack(resampled_channels)

                    self.sample_rate = target_sample_rate

                # 再转换通道数
                if target_channels and self.audio_data.shape[1] != target_channels:
                    if target_channels == 1 and self.audio_data.shape[1] > 1:
                        # 多声道转単声道：取平均值
                        self.audio_data = np.mean(self.audio_data, axis=1, keepdims=True).astype(self.audio_data.dtype)
                        self.channel_count = 1
                    elif target_channels > self.audio_data.shape[1]:
                        # 增加声道数：复制现有的声道
                        current_channels = self.audio_data.shape[1]
                        expanded_data = np.zeros((self.audio_data.shape[0], target_channels),
                                                 dtype=self.audio_data.dtype)
                        for i in range(target_channels):
                            expanded_data[:, i] = self.audio_data[:, i % current_channels]
                        self.audio_data = expanded_data
                        self.channel_count = target_channels
                    else:
                        # 减少声道数：截取前面的声道
                        self.audio_data = self.audio_data[:, :target_channels]
                        self.channel_count = target_channels

            except ImportError:
                print("Warning: Need to install resampy library for audio conversion: pip install resampy")
                print("Continuing with original audio data, may cause compatibility issues")
            except Exception as e:
                print(f"Audio conversion failed, using original data: {e}")

    def get_audio_info(self) -> dict:
        """
        获取音频文件信息

        Returns:
            dict: 包含音频信息的字典
        """
        if not self.is_loaded:
            self.load_audio()

        return {
            'file_path': str(self.file_path),
            'file_format': self.file_path.suffix.lower(),
            'sample_rate': self.sample_rate,
            'channel_count': self.channel_count,
            'duration': self.duration,
            'total_samples': len(self.audio_data) if self.audio_data is not None else 0,
            'is_loaded': self.is_loaded
        }

    def read_audio_chunk(self, num_samples: int) -> Tuple[np.ndarray, bool]:
        """
        读取音频块

        Args:
            num_samples: 要读取的样本数

        Returns:
            Tuple[np.ndarray, bool]: (音频数据, 是否到达文件末尾)
        """
        if not self.is_loaded:
            self.load_audio()

        # 检查是否到达文件末尾
        remaining_samples = len(self.audio_data) - self.current_position

        if remaining_samples <= 0:
            # 已达文件末尾，返回零数据
            return np.zeros((num_samples, self.channel_count), dtype=self.audio_data.dtype), True

        # 确定实际读取的样本数
        actual_samples = min(num_samples, remaining_samples)

        # 提取音频数据
        chunk = self.audio_data[self.current_position:self.current_position + actual_samples]

        # 更新当前位置
        self.current_position += actual_samples

        # 如果读取的样本数少于请求的样本数，在末尾填充零
        if actual_samples < num_samples:
            padding = np.zeros((num_samples - actual_samples, self.channel_count), dtype=self.audio_data.dtype)
            chunk = np.vstack([chunk, padding])
            reached_end = True
        else:
            reached_end = (self.current_position >= len(self.audio_data))

        return chunk, reached_end

    def reset_position(self):
        """重置到文件开头"""
        self.current_position = 0

    def seek(self, position: float):
        """
        跳转到指定时间位置（秒）

        Args:
            position: 时间位置（秒）
        """
        sample_position = int(position * self.sample_rate)

        if sample_position < 0:
            sample_position = 0
        elif sample_position >= len(self.audio_data):
            sample_position = len(self.audio_data) - 1

        self.current_position = sample_position

    def get_remaining_duration(self) -> float:
        """
        获取剩余持续时间

        Returns:
            float: 剩余持续时间（秒）
        """
        if not self.is_loaded:
            self.load_audio()

        remaining_samples = len(self.audio_data) - self.current_position
        return remaining_samples / self.sample_rate if self.sample_rate > 0 else 0.0

    def validate_format_compatibility(self, target_sample_rate: int, target_channels: int) -> dict:
        """
        验证格式兼容性

        Args:
            target_sample_rate: 目标采样率
            target_channels: 目标通道数

        Returns:
            dict: 兼容性验证结果
        """
        if not self.is_loaded:
            self.load_audio()

        compatibility = {
            'is_compatible': True,
            'sample_rate_match': self.sample_rate == target_sample_rate,
            'channel_match': self.channel_count == target_channels,
            'needs_conversion': False,
            'issues': []
        }

        if self.sample_rate != target_sample_rate:
            compatibility['is_compatible'] = False
            compatibility['needs_conversion'] = True
            compatibility['issues'].append(
                f"Sample rate mismatch: file {self.sample_rate}Hz, target {target_sample_rate}Hz")

        if self.channel_count != target_channels:
            compatibility['is_compatible'] = False
            compatibility['needs_conversion'] = True
            compatibility['issues'].append(
                f"Channel count mismatch: file {self.channel_count}ch, target {target_channels}ch")

        return compatibility

    def convert_audio_data(self, target_sample_rate: int, target_channels: int) -> np.ndarray:
        """
        转换音频数据以匹配目标参数

        Args:
            target_sample_rate: 目标采样率
            target_channels: 目标通道数

        Returns:
            np.ndarray: 转换后的音频数据
        """
        if not self.is_loaded:
            self.load_audio()

        audio_data = self.audio_data.copy()

        # 重采样处理
        if self.sample_rate != target_sample_rate:
            try:
                import resampy
                # 重采样到目标采样率
                if audio_data.shape[1] == 1:
                    # 単声道
                    audio_data = resampy.resample(audio_data.flatten(), self.sample_rate, target_sample_rate)
                    audio_data = audio_data.reshape(-1, 1)
                else:
                    # 多声道，逐个声道重采样
                    resampled_channels = []
                    for ch in range(audio_data.shape[1]):
                        ch_data = resampy.resample(audio_data[:, ch], self.sample_rate, target_sample_rate)
                        resampled_channels.append(ch_data)
                    # 重新组合多声道数据
                    audio_data = np.column_stack(resampled_channels)

                self.sample_rate = target_sample_rate
            except ImportError:
                raise ImportError("To perform resampling, need to install resampy library: pip install resampy")

        # 通道数调整处理
        if audio_data.shape[1] != target_channels:
            if target_channels == 1 and audio_data.shape[1] > 1:
                # 多声道转単声道：取平均值
                audio_data = np.mean(audio_data, axis=1, keepdims=True).astype(audio_data.dtype)
            elif target_channels > audio_data.shape[1]:
                # 增加声道数：复制现有的声道
                current_channels = audio_data.shape[1]
                expanded_data = np.zeros((audio_data.shape[0], target_channels), dtype=audio_data.dtype)
                for i in range(target_channels):
                    expanded_data[:, i] = audio_data[:, i % current_channels]
                audio_data = expanded_data
            else:
                # 减少声道数：截取前面的声道
                audio_data = audio_data[:, :target_channels]

        self.channel_count = target_channels

        return audio_data


def validate_audio_file_format(file_path: str) -> Tuple[bool, str]:
    """
    验证音频文件格式兼容性

    Args:
        file_path: 音频文件路径

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    from pathlib import Path
    file_path_obj = Path(file_path)
    
    # 检查文件是否存在先
    if not file_path_obj.exists():
        return False, f"音频文件不存在: {file_path}"
    
    # 检查文件格式是否支持
    if file_path_obj.suffix.lower() not in AudioFileProcessor.SUPPORTED_FORMATS:
        return False, f"无法加载 - 不支持的音频文件格式 {file_path_obj.suffix}，支持的格式: {AudioFileProcessor.SUPPORTED_FORMATS}"
    
    try:
        processor = AudioFileProcessor(file_path)
        processor.load_audio()
        return True, "文件格式有效"
    except ValueError as e:
        return False, f"无法加载音频文件: {str(e)}"
    except Exception as e:
        return False, f"无法加载音频文件: {str(e)}"
