import pyaudio
import numpy as np
import wave
import time
import os
import signal
import sys

class ReferenceChannelRecorder:
    """Utility to record only the reference channel from a multi-channel audio stream."""
    
    def __init__(self, config, output_path="reference_recording.wav"):
        self.config = config
        self.output_path = output_path
        self.sample_rate = config.sample_rate
        self.buffer_size = config.buffer_size
        self.ref_channel_idx = config.aec.reference_channel_index
        self.device_name = config.device_name
        self.device_index = config.device_index
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.wave_file = None
        self.is_running = False
        
        # Determine actual input channels from device
        self.input_channels = self._get_device_channels()
        print(f"Recorder initialized:")
        print(f"  - Target Channel Index: {self.ref_channel_idx}")
        print(f"  - Device Channels: {self.input_channels}")
        print(f"  - Output Path: {self.output_path}")

    def _get_device_channels(self):
        """Find the device and return its input channel count."""
        target_idx = self.device_index
        
        if self.device_name:
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                if self.device_name in info['name'] and info['maxInputChannels'] > 0:
                    target_idx = i
                    break
        
        if target_idx is None:
            info = self.p.get_default_input_device_info()
            target_idx = info['index']
        else:
            info = self.p.get_device_info_by_index(target_idx)
            
        self.device_index = target_idx
        return info['maxInputChannels']

    def start(self):
        """Start the recording stream."""
        if self.ref_channel_idx >= self.input_channels:
            print(f"Error: Reference channel index {self.ref_channel_idx} exceeds device channel count {self.input_channels}")
            return

        self.wave_file = wave.open(self.output_path, 'wb')
        self.wave_file.setnchannels(1) # Mono recording
        self.wave_file.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        self.wave_file.setframerate(self.sample_rate)

        def callback(in_data, frame_count, time_info, status):
            if not self.is_running:
                return (None, pyaudio.paAbort)
            
            # Extract the specific channel
            data = np.frombuffer(in_data, dtype=np.int16)
            data = data.reshape(-1, self.input_channels)
            ref_data = data[:, self.ref_channel_idx]
            
            # Write to wave file
            self.wave_file.writeframes(ref_data.tobytes())
            return (None, pyaudio.paContinue)

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.input_channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.buffer_size,
            stream_callback=callback
        )

        self.is_running = True
        self.stream.start_stream()
        print(f"Recording started on device {self.device_index}. Press Ctrl+C to stop.")

    def stop(self):
        """Stop and cleanup."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.wave_file:
            self.wave_file.close()
        self.p.terminate()
        print(f"Recording saved to {self.output_path}")

def record_reference_standalone(config_path="config.yaml", output_file="reference_recording.wav"):
    """Entry point for the recording tool."""
    from src.core.config.config_loader import load_config_from_file
    
    try:
        config = load_config_from_file(config_path)
        recorder = ReferenceChannelRecorder(config, output_file)
        
        def signal_handler(sig, frame):
            print("\nStopping recording...")
            recorder.stop()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler) # Basic signal handling for Ctrl+C
        
        recorder.start()
        
        # Keep main thread alive
        while recorder.is_running:
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Recording failed: {e}")
        if 'recorder' in locals():
            recorder.stop()
