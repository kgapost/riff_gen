"""Model configuration (configs/model/{baseline,big,huge}.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EncoderConfig:
    n_layer: int = 3
    d_model: int = 384
    n_head: int = 6
    d_ff: int = 1024


@dataclass
class ModelConfig:
    name: str = "baseline"
    vocab_size: int = 2752
    n_layer: int = 12
    d_model: int = 768
    n_head: int = 12
    d_ff: int = 2048
    max_seq_len: int = 4096
    dropout: float = 0.1
    attn_dropout: float = 0.0
    tie_embeddings: bool = True
    # positions: share of rotary dimensions per axis (token index, musical time, position in bar)
    rope_axes: dict = field(default_factory=lambda: {"token": 1.0})
    rope_theta: float = 10000.0
    time_period: tuple = (1.5, 4096.0)      # musical-time axis, in 16th notes
    phase_period: tuple = (1.5, 64.0)       # position-in-bar axis, in 16th notes
    music_features: bool = False            # pitch/fret/string/beat/... embeddings
    # hierarchical "song memory" (huge)
    hierarchical: bool = False
    bar_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    song_encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(6, 768, 12, 2048))
    cross_attn_every: int = 4
    max_bar_tokens: int = 96
    max_song_bars: int = 256
    aux_weights: dict = field(default_factory=lambda: {"pc": 0.1, "onset": 0.1, "section": 0.05, "contrastive": 0.05})
    grad_checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        for k in ("bar_encoder", "song_encoder"):
            if k in d and isinstance(d[k], dict):
                d[k] = EncoderConfig(**d[k])
        for k in ("time_period", "phase_period"):
            if k in d:
                d[k] = tuple(d[k])
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head
