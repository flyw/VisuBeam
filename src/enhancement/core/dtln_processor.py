import numpy as np
import tensorflow as tf
from ..config.dtln_config import DTLNConfig

class DTLNProcessor:
    def __init__(self, config: DTLNConfig, dtln_model=None):
        self.config = config
        self.in_buffer = np.zeros((self.config.block_len))
        self.out_buffer = np.zeros((self.config.block_len))
        self.frame_buffer = np.array([])
        if self.config.enabled:
            if dtln_model is None:
                raise ValueError("DTLN is enabled, but no model was provided.")
            self.model = dtln_model
            self.infer = self.model.signatures["serving_default"]
            
    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return frame

        self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        processed_output = np.array([], dtype=np.float32)

        while len(self.frame_buffer) >= self.config.block_shift:
            # Shift input buffer
            self.in_buffer[:-self.config.block_shift] = self.in_buffer[self.config.block_shift:]
            # Copy new data to input buffer
            self.in_buffer[-self.config.block_shift:] = self.frame_buffer[:self.config.block_shift]
            # Remove processed data from frame buffer
            self.frame_buffer = self.frame_buffer[self.config.block_shift:]

            # Create a batch dimension of one
            in_block = np.expand_dims(self.in_buffer, axis=0).astype('float32')
            
            # Process one block
            out_block = self.infer(tf.constant(in_block))['conv1d_1']
            
            # Shift output buffer
            self.out_buffer[:-self.config.block_shift] = self.out_buffer[self.config.block_shift:]
            self.out_buffer[-self.config.block_shift:] = np.zeros((self.config.block_shift))
            self.out_buffer += np.squeeze(out_block)
            
            # Append to output
            processed_output = np.concatenate((processed_output, self.out_buffer[:self.config.block_shift]))

        return processed_output

    def is_enabled(self) -> bool:
        return self.config.enabled

    def should_save_output(self) -> bool:
        return self.config.save_output