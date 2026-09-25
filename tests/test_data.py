import pytest
from src import config
from src.data import read_source, read_ground_truth

needs_data = pytest.mark.skipif(
    not (config.DATA_DIR / "train" / "train_source1.tsv").exists(), reason="dataset not present"
)

@needs_data
def test_read_source_all_strings_no_nulls():
    df = read_source("test", 1)
    assert df.columns == ["entity_id", "business_name", "business_address", "country"]
    assert df.height == 1_732_544
    assert df.null_count().sum_horizontal().item() == 0

@needs_data
def test_read_ground_truth_singletons_are_empty_sets():
    gt = read_ground_truth()
    assert len(gt) == 2_206_821
    assert sum(1 for v in gt.values() if not v) == 123_247
