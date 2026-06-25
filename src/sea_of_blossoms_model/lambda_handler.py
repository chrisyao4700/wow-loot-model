from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from sea_of_blossoms_model.baselines import DeterministicLootModel
from sea_of_blossoms_model.models import ResidueFrequencyLootModel
from sea_of_blossoms_model.serialization import (
    prediction_input_from_api,
    prediction_output_to_api,
)


def handler(event: dict[str, Any], _context: object) -> dict[str, Any]:
    payload = prediction_input_from_api(event)
    return prediction_output_to_api(_load_model().predict(payload))


@lru_cache(maxsize=1)
def _load_model() -> object:
    artifact_path = os.environ.get("MODEL_ARTIFACT_PATH")
    if artifact_path and Path(artifact_path).exists():
        return ResidueFrequencyLootModel.from_artifact_path(Path(artifact_path))

    return DeterministicLootModel()
