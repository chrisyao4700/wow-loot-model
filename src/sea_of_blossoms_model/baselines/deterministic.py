from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Callable, TypeVar

from sea_of_blossoms_model.contracts import (
    LootItem,
    LootModelMetadata,
    LootPredictionInput,
    LootPredictionOutput,
    LootTrainingInput,
    LootTrainingResult,
    BossLootPrediction,
)

_LEGENDARY_DROP_CHANCE = 0.3
_REGULAR_ITEM_COUNT = 4
_ZG_GAHZRANKA_NPC_ID = 15114
_ZG_HIDDEN_BOSS_NPC_IDS = {15082, 15083, 15084, 15085}
_ZG_SHARED_AMULET_ITEM_IDS = {19939, 19940, 19941, 19942}

T = TypeVar("T")


class DeterministicLootModel:
    metadata = LootModelMetadata(
        id="deterministic-seeded-loot",
        version="1.0.0",
        description="Seeded baseline equivalent to the original backend mock predictor.",
    )

    def train(self, payload: LootTrainingInput) -> LootTrainingResult:
        return LootTrainingResult(
            model=self.metadata,
            trained_at=datetime.now(timezone.utc).isoformat(),
            example_count=len(payload.examples),
            validation_example_count=len(payload.validation_examples),
            artifact_path=payload.artifact_path,
            metrics={"observed_runs": float(len(payload.examples))},
        )

    def predict(self, payload: LootPredictionInput) -> LootPredictionOutput:
        predictions: list[BossLootPrediction] = []

        for boss_index, boss in enumerate(payload.bosses):
            guaranteed_item_ids = {
                item.item_id for item in boss.items if item.guaranteed
            }
            random = _seeded_random(
                f"{payload.raid_id}:{payload.instance_id}:{boss.id}:{boss_index}"
            )
            predictions.append(
                BossLootPrediction(
                    boss_id=boss.id,
                    boss_name=boss.name,
                    items=tuple(
                        _pick_boss_loot(
                            random,
                            boss.items,
                            guaranteed_item_ids=guaranteed_item_ids or None,
                            npc_id=boss.npc_id,
                            legendary_drop_eligible=boss.legendary_drop_eligible,
                        )
                    ),
                )
            )

        return LootPredictionOutput(bosses=tuple(predictions), model=self.metadata)


def _pick_boss_loot(
    random: Callable[[], float],
    pool: tuple[LootItem, ...],
    *,
    guaranteed_item_ids: set[int] | None = None,
    npc_id: int | None = None,
    legendary_drop_eligible: bool | None = None,
) -> list[LootItem]:
    can_roll_legendary = legendary_drop_eligible is not False

    if npc_id == _ZG_GAHZRANKA_NPC_ID:
        return [_pick(pool, random)]

    if npc_id is not None and npc_id in _ZG_HIDDEN_BOSS_NPC_IDS:
        return _pick_zg_hidden_boss_loot(random, pool)

    guaranteed = [item for item in pool if item.item_id in (guaranteed_item_ids or set())]
    legendary_items = [
        item
        for item in pool
        if item.rarity == "legendary" and item.item_id not in (guaranteed_item_ids or set())
    ]
    filler_pool = [
        item
        for item in pool
        if item.rarity != "legendary" and item.item_id not in (guaranteed_item_ids or set())
    ]
    include_legendary = (
        can_roll_legendary
        and len(legendary_items) > 0
        and random() < _LEGENDARY_DROP_CHANCE
    )

    selected = list(guaranteed)
    regular_count = 0

    for item in _shuffle(filler_pool, random):
        if regular_count >= _REGULAR_ITEM_COUNT:
            break

        if all(existing.item_id != item.item_id for existing in selected):
            selected.append(item)
            regular_count += 1

    while regular_count < _REGULAR_ITEM_COUNT and regular_count < len(filler_pool):
        candidate = _pick(tuple(filler_pool), random)
        if all(existing.item_id != candidate.item_id for existing in selected):
            selected.append(candidate)
            regular_count += 1

    if include_legendary:
        selected.append(_pick(tuple(legendary_items), random))

    return selected


def _pick_zg_hidden_boss_loot(
    random: Callable[[], float],
    pool: tuple[LootItem, ...],
) -> list[LootItem]:
    shared_pool = tuple(item for item in pool if item.item_id in _ZG_SHARED_AMULET_ITEM_IDS)
    other_pool = tuple(item for item in pool if item.item_id not in _ZG_SHARED_AMULET_ITEM_IDS)

    if len(shared_pool) == 0 or len(other_pool) == 0:
        return _pick_boss_loot(random, pool)

    selected = [_pick(shared_pool, random) for _ in range(3)]
    selected.append(_pick(other_pool, random))
    return selected


def _pick(items: tuple[T, ...], random: Callable[[], float]) -> T:
    return items[_random_int(random, 0, len(items) - 1)]


def _shuffle(items: list[T], random: Callable[[], float]) -> list[T]:
    copy = list(items)
    for index in range(len(copy) - 1, 0, -1):
        swap_index = _random_int(random, 0, index)
        copy[index], copy[swap_index] = copy[swap_index], copy[index]
    return copy


def _random_int(random: Callable[[], float], minimum: int, maximum: int) -> int:
    return int(random() * (maximum - minimum + 1)) + minimum


def _seeded_random(seed: str) -> Callable[[], float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    state = int.from_bytes(digest[:4], "big") or 1

    def next_value() -> float:
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return state / 0x100000000

    return next_value
