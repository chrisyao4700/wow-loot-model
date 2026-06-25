from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

ItemRarity = Literal["common", "uncommon", "rare", "epic", "legendary"]


@dataclass(frozen=True)
class LootItem:
    item_id: int
    name: str
    rarity: ItemRarity


@dataclass(frozen=True)
class RaidItemDefinition(LootItem):
    guaranteed: bool = False


@dataclass(frozen=True)
class RaidBossDefinition:
    id: str
    name: str
    items: tuple[RaidItemDefinition, ...]
    npc_id: int | None = None
    legendary_drop_eligible: bool | None = None


@dataclass(frozen=True)
class BossLootPrediction:
    boss_id: str
    boss_name: str
    items: tuple[LootItem, ...]


@dataclass(frozen=True)
class LootModelMetadata:
    id: str
    version: str
    description: str | None = None


@dataclass(frozen=True)
class LootPredictionInput:
    prediction_id: str
    raid_id: int
    raid_name: str
    instance_id: str
    bosses: tuple[RaidBossDefinition, ...]


@dataclass(frozen=True)
class LootPredictionOutput:
    bosses: tuple[BossLootPrediction, ...]
    model: LootModelMetadata | None = None


@dataclass(frozen=True)
class ObservedBossLoot:
    boss_id: str
    observed_item_ids: tuple[int, ...]
    boss_name: str | None = None
    boss_index: int | None = None
    npc_id: int | None = None


@dataclass(frozen=True)
class ObservedLootRun:
    instance_id: str
    raid_id: int
    bosses: tuple[ObservedBossLoot, ...]
    raid_name: str | None = None
    observed_at: str | None = None


@dataclass(frozen=True)
class LootTrainingInput:
    examples: tuple[ObservedLootRun, ...]
    validation_examples: tuple[ObservedLootRun, ...] = ()
    artifact_path: str | None = None
    parameters: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class LootTrainingResult:
    model: LootModelMetadata
    trained_at: str
    example_count: int
    validation_example_count: int = 0
    artifact_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class LootModel(Protocol):
    metadata: LootModelMetadata

    def predict(self, payload: LootPredictionInput) -> LootPredictionOutput:
        ...


class TrainableLootModel(LootModel, Protocol):
    def train(self, payload: LootTrainingInput) -> LootTrainingResult:
        ...
