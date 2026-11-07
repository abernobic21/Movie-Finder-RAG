from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class ComponentConfig(BaseModel):
    """Selects an implementation by `type`. Other fields are passed to it as arguments."""

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    data_path: Path
    index_root: Path = Path("data/indexes")
    chunker: ComponentConfig
    embedder: ComponentConfig
    retriever: ComponentConfig
    candidate_chunks: int = 100
    llm: ComponentConfig | None = None


def load_config(path: Path) -> PipelineConfig:
    with path.open(encoding="utf-8") as f:
        return PipelineConfig.model_validate(yaml.safe_load(f))