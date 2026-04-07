from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class WebRtcApmConfig:
    """
    Configuration for WebRTC Audio Processing Module (APM).
    Allows granular control over AEC, NS, TS, AGC, etc.
    """
    enable_aec: bool = True
    enable_ns: bool = True
    ns_level: int = 1  # 0: Low, 1: Moderate, 2: High, 3: VeryHigh
    enable_ts: bool = False
    enable_agc: bool = True
    agc_mode: int = 1  # 0: AdaptiveAnalog, 1: AdaptiveDigital, 2: FixedDigital
    enable_hpf: bool = True
    enable_vad: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebRtcApmConfig':
        if not data:
            return cls()
        
        return cls(
            enable_aec=data.get('enable_aec', True),
            enable_ns=data.get('enable_ns', True),
            ns_level=data.get('noise_suppression_level', data.get('ns_level', 1)),
            enable_ts=data.get('enable_ts', False),
            enable_agc=data.get('enable_agc', True),
            agc_mode=data.get('gain_control_mode', data.get('agc_mode', 1)),
            enable_hpf=data.get('enable_hpf', True),
            enable_vad=data.get('enable_vad', True)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
