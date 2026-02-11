"""Tests for data_store module."""
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from lib.data_store import WarReportStore


@pytest.fixture
def tmp_data_dir():
    """Create a temporary directory for test data."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def store(tmp_data_dir):
    """Create a WarReportStore with temp directory."""
    return WarReportStore(tmp_data_dir)


class TestInit:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "subdir" / "data"
            store = WarReportStore(str(path))
            assert path.exists()

    def test_paths_set(self, store, tmp_data_dir):
        assert store.data_dir == Path(tmp_data_dir)
        assert store.reports_path == Path(tmp_data_dir) / "war_reports.parquet"
        assert store.features_path == Path(tmp_data_dir) / "narrative_features.parquet"


class TestReports:
    def test_load_empty(self, store):
        df = store.load_reports()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_save_and_load(self, store):
        df = pd.DataFrame({"file": ["a.txt", "b.txt"], "content": ["hello", "world"]})
        store.save_reports(df)

        loaded = store.load_reports()
        assert len(loaded) == 2
        assert list(loaded.columns) == ["file", "content"]

    def test_append_deduplicates(self, store):
        df1 = pd.DataFrame({"file": ["a.txt", "b.txt"], "val": [1, 2]})
        store.save_reports(df1)

        df2 = pd.DataFrame({"file": ["b.txt", "c.txt"], "val": [2, 3]})
        store.append_reports(df2)

        loaded = store.load_reports()
        assert len(loaded) == 3  # a, b, c (b not duplicated)

    def test_append_to_empty(self, store):
        df = pd.DataFrame({"file": ["x.txt"], "val": [99]})
        store.append_reports(df)

        loaded = store.load_reports()
        assert len(loaded) == 1


class TestFeatures:
    def test_load_empty(self, store):
        df = store.load_features()
        assert len(df) == 0

    def test_save_and_load(self, store):
        df = pd.DataFrame({"narrative": ["test text"], "Narratives_sim_top_tag": ["MOE"]})
        store.save_features(df)

        loaded = store.load_features()
        assert len(loaded) == 1
        assert loaded.iloc[0]["Narratives_sim_top_tag"] == "MOE"


class TestProcessingLog:
    def test_initially_empty(self, store):
        assert store.get_processed_directories() == set()

    def test_log_and_retrieve(self, store):
        store.log_processed_directory("/data/dir1", file_count=5)
        store.log_processed_directory("/data/dir2", file_count=3)

        processed = store.get_processed_directories()
        assert "/data/dir1" in processed
        assert "/data/dir2" in processed

    def test_unprocessed_directories(self, store, tmp_data_dir):
        # Create subdirectories
        base = Path(tmp_data_dir) / "war_files"
        (base / "batch1").mkdir(parents=True)
        (base / "batch2").mkdir(parents=True)
        (base / "batch3").mkdir(parents=True)

        # Mark batch1 as processed
        store.log_processed_directory(str(base / "batch1"))

        unprocessed = store.get_unprocessed_directories(str(base))
        unprocessed_names = [Path(p).name for p in unprocessed]
        assert "batch2" in unprocessed_names
        assert "batch3" in unprocessed_names
        assert "batch1" not in unprocessed_names


class TestQuery:
    def test_sql_query(self, store):
        df = pd.DataFrame({"file": ["a.txt", "b.txt"], "size": [100, 200]})
        store.save_reports(df)

        result = store.query("SELECT * FROM reports WHERE size > 150")
        assert len(result) == 1
        assert result.iloc[0]["file"] == "b.txt"

    def test_query_reports_convenience(self, store):
        df = pd.DataFrame({"file": ["a.txt", "b.txt"], "size": [100, 200]})
        store.save_reports(df)

        result = store.query_reports(where="size > 150")
        assert len(result) == 1


class TestMoeMop:
    def test_get_moe_mop_reports(self, store):
        df = pd.DataFrame({
            "narrative": ["text1", "text2", "text3"],
            "Narratives_sim_top_tag": ["MOE", "MOP", "OTHER"],
        })
        store.save_features(df)

        result = store.get_moe_mop_reports()
        assert len(result) == 2


class TestStatus:
    def test_status_empty(self, store):
        status = store.status()
        assert status['reports_exist'] is False
        assert status['features_exist'] is False
        assert status['processed_directories'] == 0

    def test_status_with_data(self, store):
        store.save_reports(pd.DataFrame({"file": ["a"]}))
        status = store.status()
        assert status['reports_exist'] is True
        assert status['report_count'] == 1


class TestRepr:
    def test_repr(self, store):
        r = repr(store)
        assert "WarReportStore" in r
        assert "reports=0" in r
