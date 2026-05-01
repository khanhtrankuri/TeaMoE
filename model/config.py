from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ExpertGroupConfig:
    """Cấu hình cho 1 nhóm expert (5 expert/nhóm)"""
    group_id: int
    group_name: str
    num_experts: int = 5
    expert_dim: int = 1024
    ff_multiplier: int = 4
    dropout: float = 0.1
    # Các lĩnh vực chuyên biệt
    specialization: str = "vowels"  # vowels, plosives, fricatives, nasals, male, female, clean, other


@dataclass
class ModelConfig:
    """Cấu hình tổng thể cho MoE-Conformer + RNN-T"""
    # Conformer Encoder
    num_layers: int = 24
    moe_start_layer: int = 6
    moe_end_layer: int = 18
    model_dim: int = 1024
    num_heads: int = 16
    conv_kernel_size: int = 31
    ff_multiplier: int = 4

    # Gating Network
    num_groups: int = 8  # 8 nhóm chuyên biệt
    experts_per_group: int = 5  # 5 expert/nhóm
    total_experts: int = 40  # 8 * 5
    top_k_groups: int = 1  # Chọn 1 nhóm (5 expert) mỗi frame
    top_k_inference: int = 2  # Chọn 2 expert tốt nhất mỗi nhóm khi inference

    # Expert Groups Configuration
    group_configs: List[ExpertGroupConfig] = None

    # RNN-T Decoder
    vocab_size: int = 5000  # LibriSpeech vocab size
    decoder_hidden: int = 1024
    decoder_layers: int = 2
    blank_id: int = 0

    # Competition & Distillation
    competition_freq_steps: int = 1000  # Chạy cạnh tranh mỗi 1000 batch
    distillation_weight: float = 0.1
    load_balance_weight: float = 0.01
    z_loss_weight: float = 0.001
    ctc_phone_weight: float = 0.3

    # Audio
    sample_rate: int = 16000
    n_mels: int = 80
    hop_length: int = 256
    win_length: int = 1024

    # Competition JAX params
    alpha: float = 0.5  # Cho thuật toán natural niches
    use_matchmaker: bool = True

    def __post_init__(self):
        if self.group_configs is None:
            self.group_configs = [
                ExpertGroupConfig(0, "vowels", specialization="vowels"),
                ExpertGroupConfig(1, "plosives", specialization="plosives"),
                ExpertGroupConfig(2, "fricatives", specialization="fricatives"),
                ExpertGroupConfig(3, "nasals", specialization="nasals"),
                ExpertGroupConfig(4, "male_speakers", specialization="male"),
                ExpertGroupConfig(5, "female_speakers", specialization="female"),
                ExpertGroupConfig(6, "clean_audio", specialization="clean"),
                ExpertGroupConfig(7, "other_audio", specialization="other"),
            ]
