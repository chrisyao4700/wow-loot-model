from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sea_of_blossoms_model.contracts import (
    BossLootPrediction,
    LootItem,
    LootModelMetadata,
    LootPredictionInput,
    LootPredictionOutput,
    LootTrainingInput,
    LootTrainingResult,
    RaidBossDefinition,
)

_SCHEMA_VERSION = 1
_DEFAULT_PERIOD = 97


class ResidueFrequencyLootModel:
    def __init__(self, artifact: dict[str, Any] | None = None) -> None:
        self._artifact = artifact
        self.metadata = _metadata_from_artifact(artifact) if artifact else LootModelMetadata(
            id="residue-frequency",
            version="0.1.0",
            description="Observed-loot frequency model keyed by instanceId residue.",
        )

    @classmethod
    def from_artifact_path(cls, path: Path) -> "ResidueFrequencyLootModel":
        return cls(json.loads(path.read_text()))

    def train(self, payload: LootTrainingInput) -> LootTrainingResult:
        period = int(payload.parameters.get("period", _DEFAULT_PERIOD))
        trained_at = datetime.now(timezone.utc).isoformat()
        stats: dict[str, Any] = {}
        observed_bosses = 0
        observed_items = 0

        for example in payload.examples:
            residue = _instance_residue(example.instance_id, period)
            for boss in example.bosses:
                observed_bosses += 1
                observed_items += len(boss.observed_item_ids)
                boss_key = _boss_key(example.raid_id, boss.boss_id)
                boss_stats = stats.setdefault(
                    boss_key,
                    {
                        "dropCountCounts": {},
                        "fallbackByDraw": {},
                        "residueByDraw": {},
                    },
                )
                _increment(boss_stats["dropCountCounts"], str(len(boss.observed_item_ids)))

                residue_stats = boss_stats["residueByDraw"].setdefault(str(residue), {})
                for draw_index, item_id in enumerate(boss.observed_item_ids):
                    draw_key = str(draw_index)
                    _increment(
                        boss_stats["fallbackByDraw"].setdefault(draw_key, {}),
                        str(item_id),
                    )
                    _increment(
                        residue_stats.setdefault(draw_key, {}),
                        str(item_id),
                    )

        artifact = {
            "schemaVersion": _SCHEMA_VERSION,
            "model": asdict(self.metadata),
            "trainedAt": trained_at,
            "parameters": {
                "period": period,
            },
            "stats": stats,
        }
        self._artifact = artifact

        if payload.artifact_path:
            artifact_path = Path(payload.artifact_path)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))

        return LootTrainingResult(
            model=self.metadata,
            trained_at=trained_at,
            example_count=len(payload.examples),
            validation_example_count=len(payload.validation_examples),
            artifact_path=payload.artifact_path,
            metrics={
                "observed_bosses": float(observed_bosses),
                "observed_items": float(observed_items),
                "period": float(period),
                "boss_stat_count": float(len(stats)),
            },
        )

    def predict(self, payload: LootPredictionInput) -> LootPredictionOutput:
        if not self._artifact:
            raise ValueError("ResidueFrequencyLootModel requires a trained artifact before prediction.")

        period = int(self._artifact["parameters"]["period"])
        residue = _instance_residue(payload.instance_id, period)
        stats = self._artifact.get("stats", {})
        predictions: list[BossLootPrediction] = []

        for boss in payload.bosses:
            boss_stats = stats.get(_boss_key(payload.raid_id, boss.id))
            predictions.append(
                BossLootPrediction(
                    boss_id=boss.id,
                    boss_name=boss.name,
                    items=tuple(_predict_boss_items(boss, boss_stats, residue)),
                )
            )

        return LootPredictionOutput(bosses=tuple(predictions), model=self.metadata)


def _predict_boss_items(
    boss: RaidBossDefinition,
    boss_stats: dict[str, Any] | None,
    residue: int,
) -> list[LootItem]:
    if not boss.items:
        return []

    item_by_id = {item.item_id: item for item in boss.items}
    if not boss_stats:
        return list(boss.items[: min(4, len(boss.items))])

    drop_count = _most_common_int(boss_stats.get("dropCountCounts", {})) or min(4, len(boss.items))
    residue_stats = boss_stats.get("residueByDraw", {}).get(str(residue), {})
    fallback_stats = boss_stats.get("fallbackByDraw", {})
    selected: list[LootItem] = []

    for draw_index in range(drop_count):
        draw_key = str(draw_index)
        item_id = _best_item_id(residue_stats.get(draw_key, {}), item_by_id)
        if item_id is None:
            item_id = _best_item_id(fallback_stats.get(draw_key, {}), item_by_id)

        if item_id is None:
            selected.append(boss.items[draw_index % len(boss.items)])
        else:
            selected.append(item_by_id[item_id])

    return selected


def _metadata_from_artifact(artifact: dict[str, Any]) -> LootModelMetadata:
    model = artifact.get("model", {})
    return LootModelMetadata(
        id=str(model.get("id", "residue-frequency")),
        version=str(model.get("version", "0.1.0")),
        description=model.get("description"),
    )


def _best_item_id(counts: dict[str, int], item_by_id: dict[int, LootItem]) -> int | None:
    valid_counts = [
        (int(item_id), count)
        for item_id, count in counts.items()
        if int(item_id) in item_by_id
    ]
    if not valid_counts:
        return None

    return sorted(valid_counts, key=lambda entry: (-entry[1], entry[0]))[0][0]


def _most_common_int(counts: dict[str, int]) -> int | None:
    if not counts:
        return None

    return int(sorted(counts.items(), key=lambda entry: (-entry[1], int(entry[0])))[0][0])


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _boss_key(raid_id: int, boss_id: str) -> str:
    return f"{raid_id}|{boss_id}"


def _instance_residue(instance_id: str, period: int) -> int:
    try:
        return int(instance_id) % period
    except ValueError:
        return sum(instance_id.encode("utf-8")) % period
