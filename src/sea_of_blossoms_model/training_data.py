from __future__ import annotations

from sea_of_blossoms_model.baselines import DeterministicLootModel
from sea_of_blossoms_model.catalog import CatalogRaid
from sea_of_blossoms_model.contracts import (
    LootPredictionInput,
    ObservedBossLoot,
    ObservedLootRun,
)


def generate_deterministic_observations(
    raids: tuple[CatalogRaid, ...],
    *,
    instance_start: int,
    instance_count: int,
    instance_step: int = 1,
) -> tuple[ObservedLootRun, ...]:
    model = DeterministicLootModel()
    examples: list[ObservedLootRun] = []

    for offset in range(instance_count):
        instance_id = str(instance_start + offset * instance_step)
        for raid in raids:
            prediction = model.predict(
                LootPredictionInput(
                    prediction_id=f"sample-{raid.numeric_id}-{instance_id}",
                    raid_id=raid.numeric_id,
                    raid_name=raid.name,
                    instance_id=instance_id,
                    bosses=raid.bosses,
                )
            )
            examples.append(
                ObservedLootRun(
                    instance_id=instance_id,
                    raid_id=raid.numeric_id,
                    raid_name=raid.name,
                    bosses=tuple(
                        ObservedBossLoot(
                            boss_id=boss_prediction.boss_id,
                            boss_name=boss_prediction.boss_name,
                            boss_index=boss_index,
                            observed_item_ids=tuple(
                                item.item_id for item in boss_prediction.items
                            ),
                        )
                        for boss_index, boss_prediction in enumerate(prediction.bosses)
                    ),
                )
            )

    return tuple(examples)
