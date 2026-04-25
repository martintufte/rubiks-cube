from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing import TypeVar

import attrs

from rubiks_cube.beam_search.solver import CompiledStep
from rubiks_cube.transform.pipeline import Pipeline

if TYPE_CHECKING:
    from pathlib import Path

    import cattrs

T = TypeVar("T")

SCHEMA_VERSION = 1


def _save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "data": data}
    path.write_text(json.dumps(payload, indent=2))


def _load_json(path: Path) -> object:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError(
            f"Missing schema_version in {path}. "
            "The file may be from an older version. Delete it and rebuild."
        )
    stored = payload["schema_version"]
    if stored != SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch in {path}: "
            f"expected {SCHEMA_VERSION}, got {stored}. "
            "Delete the file and rebuild."
        )
    return payload["data"]


@attrs.frozen
class ResourceHandler:
    """Manage resources related to solving permutation search problems."""

    resource_dir: Path
    converter: cattrs.Converter

    def __attrs_post_init__(self) -> None:
        try:
            self.resource_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to create session directory {self.resource_dir}") from exc

    @property
    def config_path(self) -> Path:
        return self.resource_dir / "config.json"

    def save_config(self, config: object) -> None:
        _save_json(self.config_path, self.converter.unstructure(config))

    def load_config(self, config_type: type[T]) -> T:
        data = _load_json(self.config_path)
        return self.converter.structure(data, config_type)

    @property
    def pipeline_path(self) -> Path:
        return self.resource_dir / "preprocess_pipeline.json"

    def save_preprocess_pipeline(self, pipeline: Pipeline) -> None:
        _save_json(self.pipeline_path, self.converter.unstructure(pipeline))

    def load_preprocess_pipeline(self) -> Pipeline:
        data = _load_json(self.pipeline_path)
        return self.converter.structure(data, Pipeline)

    @property
    def step_contexts_path(self) -> Path:
        return self.resource_dir / "step_contexts.json"

    def save_step_contexts(self, contexts: list[CompiledStep]) -> None:
        _save_json(self.step_contexts_path, self.converter.unstructure(contexts))

    def load_step_contexts(self) -> list[CompiledStep]:
        data = _load_json(self.step_contexts_path)
        return self.converter.structure(data, list[CompiledStep])
