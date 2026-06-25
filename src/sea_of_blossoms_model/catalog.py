from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sea_of_blossoms_model.contracts import RaidBossDefinition, RaidItemDefinition


@dataclass(frozen=True)
class CatalogRaid:
    numeric_id: int
    catalog_id: str
    name: str
    bosses: tuple[RaidBossDefinition, ...]


def load_catalog(path: Path) -> tuple[CatalogRaid, ...]:
    payload = json.loads(path.read_text())
    return tuple(
        _raid_from_json(index, raid)
        for index, raid in enumerate(payload.get("raids", ()), start=1)
    )


def _raid_from_json(index: int, payload: dict[str, Any]) -> CatalogRaid:
    return CatalogRaid(
        numeric_id=index,
        catalog_id=str(payload["id"]),
        name=str(payload["name"]),
        bosses=tuple(_boss_from_json(boss) for boss in payload.get("bosses", ())),
    )


def _boss_from_json(payload: dict[str, Any]) -> RaidBossDefinition:
    return RaidBossDefinition(
        id=str(payload["id"]),
        name=str(payload["name"]),
        npc_id=int(payload["npcId"]) if payload.get("npcId") is not None else None,
        legendary_drop_eligible=payload.get("legendaryDropEligible"),
        items=tuple(_item_from_json(item) for item in payload.get("items", ())),
    )


def _item_from_json(payload: dict[str, Any]) -> RaidItemDefinition:
    return RaidItemDefinition(
        item_id=int(payload["itemId"]),
        name=str(payload["name"]),
        rarity=payload["rarity"],
        guaranteed=bool(payload.get("guaranteed", False)),
    )
