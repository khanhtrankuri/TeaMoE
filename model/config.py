from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ExpertGroupConfig:
    group_id: int
    group_name: str
    num_experts: int = 5
    expert_dim: int = 1024
    ff_multiplier: int = 4
    dropout: float = 0.1
    specialization: str = "vowels"

@dataclass
class ModelConfig:
    num_layers: int = 24
    moe_start_layer: int = 6
    moe_end_layer: int = 18
    model_dim: int = 1024
    num_heads: int = 16
    conv_kernel_size: int = 31
    ff_multiplier: int = 4

    num_groups: int = 8
    experts_per_group: int = 5
    total_experts: int = 40
    top_k_groups: int = 1
    top_k_inference: int = 2

    group_configs: List[ExpertGroupConfig] = None

    vocab_size: int = 5000
    decoder_hidden: int = 1024
    decoder_layers: int = 2
    blank_id: int = 0

    competition_freq_steps: int = 1000
    distillation_weight: float = 0.1
    load_balance_weight: float = 0.01
    z_loss_weight: float = 0.001
    ctc_phone_weight: float = 0.3

    sample_rate: int = 16000
    n_mels: int = 80
    hop_length: int = 256
    win_length: int = 1024

    alpha: float = 0.5
    use_matchmaker: bool = True

    def __post_init__(self):
        if self.group_configs is None:
            self.group_configs = [
                ExpertGroupConfig(0, "vowels", expert_dim=self.model_dim, specialization="vowels"),
                ExpertGroupConfig(1, "plosives", expert_dim=self.model_dim, specialization="plosives"),
                ExpertGroupConfig(2, "fricatives", expert_dim=self.model_dim, specialization="fricatives"),
                ExpertGroupConfig(3, "nasals", expert_dim=self.model_dim, specialization="nasals"),
                ExpertGroupConfig(4, "male_speakers", expert_dim=self.model_dim, specialization="male"),
                ExpertGroupConfig(5, "female_speakers", expert_dim=self.model_dim, specialization="female"),
                ExpertGroupConfig(6, "clean_audio", expert_dim=self.model_dim, specialization="clean"),
                ExpertGroupConfig(7, "other_audio", expert_dim=self.model_dim, specialization="other"),
            ]
