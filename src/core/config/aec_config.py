from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AecConfig:
    """
    Acoustic Echo Cancellation (AEC) Configuration
    Optimized for High-Precision DOA Oriented AEC v3.0
    """
    enabled: bool = False
    reference_channel_index: Optional[int] = None
    
    # High-Precision DOA Oriented AEC v3.0 Parameters
    frame_size_ms: int = 10
    fft_size: int = 256
    filter_blocks: int = 12
    ref_buffer_size: int = 8000
    dtd_threshold: float = 0.75
    default_latency_ms: int = 120
    
    # Legacy parameter
    system_delay_ms: int = 10

    def __post_init__(self):
        if self.enabled and self.reference_channel_index is None:
            pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AecConfig':
        return cls(
            enabled=data.get('enabled', False),
            reference_channel_index=data.get('reference_channel_index'),
            frame_size_ms=data.get('frame_size_ms', 10),
            fft_size=data.get('fft_size', 256),
            filter_blocks=data.get('filter_blocks', 12),
            ref_buffer_size=data.get('ref_buffer_size', 8000),
            dtd_threshold=data.get('dtd_threshold', 0.75),
            default_latency_ms=data.get('default_latency_ms', 120),
            system_delay_ms=data.get('system_delay_ms', 0)
        )