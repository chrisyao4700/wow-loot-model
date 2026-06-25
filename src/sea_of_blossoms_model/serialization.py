from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sea_of_blossoms_model.contracts import (
    BossLootPrediction,
    LootItem,
    LootPredictionInput,
    LootPredictionOutput,
    LootTrainingInput,
    ObservedBossLoot,
    ObservedLootRun,
    RaidBossDefinition,
    RaidItemDefinition,
)


def prediction_input_from_api(payload: dict[str, Any]) -> LootPredictionInput:
    return LootPredictionInput(
        prediction_id=str(payload["predictionId"]),
        raid_id=int(payload["raidId"]),
        raid_name=str(payload["raidName"]),
        instance_id=str(payload["instanceId"]),
        bosses=tuple(_boss_from_api(boss) for boss in payload["bosses"]),
    )


def prediction_output_to_api(output: LootPredictionOutput) -> dict[str, Any]:
    return {
        "bosses": [_boss_prediction_to_api(boss) for boss in output.bosses],
        "model": asdict(output.model) if output.model else None,
    }


def training_input_from_json(payload: dict[str, Any]) -> LootTrainingInput:
    return LootTrainingInput(
        examples=tuple(_observed_run_from_json(run) for run in payload.get("examples", ())),
        validation_examples=tuple(
            _observed_run_from_json(run) for run in payload.get("validationExamples", ())
        ),
        artifact_path=payload.get("artifactPath"),
        parameters=dict(payload.get("parameters", {})),
    )


def training_input_to_json(payload: LootTrainingInput) -> dict[str, Any]:
    return {
        "examples": [_observed_run_to_json(run) for run in payload.examples],
        "validationExamples": [
            _observed_run_to_json(run) for run in payload.validation_examples
        ],
        "artifactPath": payload.artifact_path,
        "parameters": payload.parameters,
    }


def _boss_from_api(payload: dict[str, Any]) -> RaidBossDefinition:
    return RaidBossDefinition(
        id=str(payload["id"]),
        name=str(payload["name"]),
        npc_id=int(payload["npcId"]) if payload.get("npcId") is not None else None,
        legendary_drop_eligible=payload.get("legendaryDropEligible"),
        items=tuple(_item_definition_from_api(item) for item in payload["items"]),
    )


def _observed_run_from_json(payload: dict[str, Any]) -> ObservedLootRun:
    return ObservedLootRun(
        instance_id=str(payload["instanceId"]),
        raid_id=int(payload["raidId"]),
        raid_name=payload.get("raidName"),
        observed_at=payload.get("observedAt"),
        bosses=tuple(_observed_boss_from_json(boss) for boss in payload.get("bosses", ())),
    )


def _observed_run_to_json(run: ObservedLootRun) -> dict[str, Any]:
    return {
        "instanceId": run.instance_id,
        "raidId": run.raid_id,
        "raidName": run.raid_name,
        "observedAt": run.observed_at,
        "bosses": [_observed_boss_to_json(boss) for boss in run.bosses],
    }


def _observed_boss_from_json(payload: dict[str, Any]) -> ObservedBossLoot:
    return ObservedBossLoot(
        boss_id=str(payload["bossId"]),
        boss_name=payload.get("bossName"),
        boss_index=int(payload["bossIndex"]) if payload.get("bossIndex") is not None else None,
        npc_id=int(payload["npcId"]) if payload.get("npcId") is not None else None,
        observed_item_ids=tuple(int(item_id) for item_id in payload.get("observedItemIds", ())),
    )


def _observed_boss_to_json(boss: ObservedBossLoot) -> dict[str, Any]:
    return {
        "bossId": boss.boss_id,
        "bossName": boss.boss_name,
        "bossIndex": boss.boss_index,
        "npcId": boss.npc_id,
        "observedItemIds": list(boss.observed_item_ids),
    }


def _item_definition_from_api(payload: dict[str, Any]) -> RaidItemDefinition:
    return RaidItemDefinition(
        item_id=int(payload["itemId"]),
        name=str(payload["name"]),
        rarity=payload["rarity"],
        guaranteed=bool(payload.get("guaranteed", False)),
    )


def _boss_prediction_to_api(prediction: BossLootPrediction) -> dict[str, Any]:
    return {
        "bossId": prediction.boss_id,
        "bossName": prediction.boss_name,
        "items": [_item_to_api(item) for item in prediction.items],
    }


def _item_to_api(item: LootItem) -> dict[str, Any]:
    return {
        "itemId": item.item_id,
        "name": item.name,
        "rarity": item.rarity,
    }
