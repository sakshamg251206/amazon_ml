from competition.decide import write_id_lists


def test_writer_emits_every_s1_once(tmp_path):
    p = tmp_path / "m.tsv"
    write_id_lists(p, "matched_entity_ids", ["S1-1", "S1-2", "S1-3"],
                   {"S1-1": ["S3-9", "S2-4", "S2-4"], "S1-9": ["S2-1"]})
    assert p.read_text(encoding="utf-8").splitlines() == [
        "source1_entity_id\tmatched_entity_ids",
        "S1-1\tS2-4,S3-9",
        "S1-2\t",
        "S1-3\t",
    ]
