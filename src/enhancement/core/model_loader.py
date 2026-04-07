import tensorflow as tf
import logging
import queue
import threading
from typing import Optional
from src.enhancement.config.dtln_config import DTLNConfig

_logger = logging.getLogger(__name__)

class DTLNModelManager:
    """
    Manages a pool of DTLN model instances to support low-latency acquisition
    and state isolation in multi-user environments.
    """
    def __init__(self, config: DTLNConfig, max_size: int = 10):
        self._config = config
        self._max_size = max_size
        self._pool = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._created_count = 0
        _logger.info(f"DTLNModelManager initialized with a max pool size of {max_size}.")

    def preload(self, num_instances: int):
        """
        Pre-loads a specified number of model instances into the pool.
        """
        instances_created = 0
        for _ in range(num_instances):
            if self._pool.full():
                _logger.warning("DTLN model pool is full, cannot pre-load more instances.")
                break
            model = self._create_model()
            self._pool.put(model)
            instances_created += 1
        _logger.info(f"Pre-loaded {instances_created} new instances. Current pool size: {self._pool.qsize()} (total created: {self._created_count}).")

    def acquire(self, timeout: float = 5.0):
        """
        Acquires a model instance from the pool.
        
        NOTE: State isolation is achieved because each user session creates
        a new DTLNProcessor instance with fresh in_buffer/out_buffer/frame_buffer.
        The TF SavedModel itself is stateless (uses signature-based inference),
        so multiple users can safely share the same model instance.
        """
        _logger.debug(f"Attempting to acquire DTLN model. Current pool size: {self._pool.qsize()}.")
        try:
            # Try to get from pool
            model = self._pool.get(block=True, timeout=0.1) # Short timeout to check pool
            _logger.debug(f"Acquired existing DTLN model from pool. Remaining in pool: {self._pool.qsize()}.")
        except queue.Empty:
            # If pool is empty, check if we can create more
            with self._lock:
                if self._created_count < self._max_size:
                    _logger.info(f"Pool empty, creating new DTLN model instance (total created: {self._created_count + 1}).")
                    model = self._create_model()
                else:
                    _logger.warning(f"Pool empty and max size reached ({self._max_size}), waiting for a model to be released.")
                    # Wait for a released model
                    try:
                        model = self._pool.get(block=True, timeout=timeout)
                        _logger.debug(f"Acquired released DTLN model from pool after waiting. Remaining in pool: {self._pool.qsize()}.")
                    except queue.Empty:
                        _logger.error(f"Timeout ({timeout}s) waiting for DTLN model instance. No models available.")
                        raise RuntimeError("Timeout waiting for DTLN model instance")

        return model

    def release(self, model: tf.keras.Model):
        """
        Returns a model instance to the pool.
        """
        try:
            self._pool.put(model, block=False)
            _logger.debug(f"Released DTLN model to pool. Current pool size: {self._pool.qsize()}.")
        except queue.Full:
            _logger.warning(f"DTLN model pool is full, discarding returned instance. Current pool size: {self._pool.qsize()}.")

    def _create_model(self) -> tf.keras.Model:
        """Loads a new model instance from disk."""
        _logger.info(f"Loading new DTLN model instance from {self._config.model_path}")
        model = tf.saved_model.load(self._config.model_path)
        with self._lock:
            self._created_count += 1
        return model

# Global Singleton
_manager: Optional[DTLNModelManager] = None

def initialize_manager(config: DTLNConfig, max_size: int = None):
    global _manager
    if _manager is None:
        if max_size is None:
            max_size = config.pool_size
        _manager = DTLNModelManager(config, max_size)

def get_manager() -> DTLNModelManager:
    global _manager
    if _manager is None:
        raise RuntimeError("DTLNModelManager not initialized. Call initialize_manager first.")
    return _manager
