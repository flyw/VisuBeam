
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class DTLNConfig:
    enabled: bool = False
    save_output: bool = False
    model_path: str = 'dtln_saved_model'
    block_len: int = 512
    block_shift: int = 128
    pool_size: int = 4

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> 'DTLNConfig':
        return DTLNConfig(
            enabled=config.get('enabled', config.get('enable_dtln', False)),
            save_output=config.get('save_output', config.get('save_dtln_output', False)),
            model_path=config.get('model_path', 'dtln_saved_model'),
            block_len=config.get('block_len', 512),
            block_shift=config.get('block_shift', 128),
            pool_size=config.get('pool_size', 4)
        )
