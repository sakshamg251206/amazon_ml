"""Final decisions -> submission files."""
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def write_id_lists(path: Path, list_col: str, s1_ids: Sequence[str],
                   mapping: Mapping[str, Iterable[str]]) -> None:
    """One row per S1 in s1_ids (so no entity is ever missing), deduped + sorted ID lists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{list_col}\n")
        for s1 in s1_ids:
            f.write(f"{s1}\t{','.join(sorted(set(mapping.get(s1, ()))))}\n")
