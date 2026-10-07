"""Validated local model configuration."""

from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GenerationModel(ConfigModel):
    id: str
    path: Path
    served_name: str
    endpoint: str
    dtype: str = "bfloat16"
    max_model_len: int = Field(ge=1024, le=262144)
    max_num_seqs: int = Field(ge=1, le=32)
    gpu_memory_utilization: float = Field(gt=0, le=0.95)
    tool_parser: str
    reasoning_parser: str | None = None

    @field_validator("path", mode="before")
    @classmethod
    def parse_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("model path must be absolute")
        return path

    @field_validator("endpoint")
    @classmethod
    def only_loopback(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
            raise ValueError("model endpoint must be a loopback HTTP URL with explicit port")
        if parsed.hostname != "localhost" and not ip_address(parsed.hostname).is_loopback:
            raise ValueError("model endpoint must be loopback")
        return value.rstrip("/")


class RetrievalModel(ConfigModel):
    id: str
    path: Path
    dimension: int | None = None

    @field_validator("path", mode="before")
    @classmethod
    def parse_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("model path must be absolute")
        return path


class Profile(ConfigModel):
    generation_device: int | None
    retrieval_device: int | str


class ModelConfig(ConfigModel):
    generation: dict[str, GenerationModel]
    embedding: RetrievalModel
    reranker: RetrievalModel
    profiles: dict[str, Profile]


def load_model_config(path: Path) -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
