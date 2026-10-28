#!/usr/bin/env python3
"""离线加载/路径映射回归测试；FakeEvaluator 仅测入口连通，不测模型性能。"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
import portable_data
from ctrl_data_common import load_image
from trajectory_runtime import candidate_id
from generator_adapter import image_to_data_uri


class PortableDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(mkdtemp(prefix="portable-data-test-", dir=HERE / "assets"))
        cls.benchmark = cls.root / "assets/benchmark"
        cls.benchmark.mkdir(parents=True)
        cls.query = cls.benchmark / "query_images/eb_42.jpg"
        cls.query.parent.mkdir()
        cls.corpus = cls.root / "assets/corpus_images/example.jpg"
        cls.corpus.parent.mkdir()
        Image.new("RGB", (20, 12), "white").save(cls.query)
        Image.new("RGB", (24, 18), "red").save(cls.corpus)
        cls.row = {"index": 42, "question": "测试问题", "A": "甲", "B": "乙", "answer": "A",
                   "query_image_path": "query_images/eb_42.jpg"}
        cls.source = cls.benchmark / "endobench_test.jsonl"
        cls.source.write_text(json.dumps(cls.row, ensure_ascii=False) + "\n", encoding="utf-8")
        cls.old_path = "/不存在的原机器/证据图.jpg"
        (cls.benchmark / "image_path_map.json").write_text(json.dumps(
            {cls.old_path: "assets/corpus_images/example.jpg"}, ensure_ascii=False), encoding="utf-8")

    def setUp(self):
        self.patch = patch.object(portable_data, "PKGROOT", self.root)
        self.patch.start()
        portable_data.image_path_map.cache_clear()

    def tearDown(self):
        self.patch.stop()
        portable_data.image_path_map.cache_clear()

    def test_local_query_path_resolves_after_relocation(self):
        rows = portable_data.load_local_benchmark(self.source)
        self.assertEqual(rows[0]["query_image_path"], str(self.query))
        self.assertEqual(rows[0]["answer"], "A")

    def test_evidence_load_preserves_candidate_identity(self):
        candidate = {"sample_id": "same-id", "image_path": self.old_path}
        identity = candidate_id(candidate)
        self.assertEqual(load_image(candidate["image_path"], 128).size, (24, 18))
        self.assertEqual(candidate["image_path"], self.old_path)
        self.assertEqual(candidate_id(candidate), identity)
        self.assertTrue(image_to_data_uri(self.old_path).startswith("data:image/jpeg;base64,"))

    def test_missing_image_raises(self):
        with self.assertRaises(FileNotFoundError):
            portable_data.resolve_image_path("/不存在的原机器/缺失.jpg")
        bad = self.benchmark / "bad.jsonl"
        bad.write_text(json.dumps({**self.row, "query_image_path": "missing.jpg"}) + "\n")
        with self.assertRaises(FileNotFoundError):
            portable_data.load_local_benchmark(bad)

    def test_duplicate_index_rejected(self):
        bad = self.benchmark / "duplicate.jsonl"
        bad.write_text((json.dumps(self.row) + "\n") * 2)
        with self.assertRaises(ValueError):
            portable_data.load_local_benchmark(bad)

    def test_cli_uses_index_not_position_and_never_downloads_dataset(self):
        spec = importlib.util.spec_from_file_location("offline_eval_under_test", HERE / "eval_endobench.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        selected = self.benchmark / "selected.jsonl"
        selected.write_text('{"qid": "eb_42"}\n')

        class FakeEvaluator:
            def __init__(self, args):
                pass

            def run_sample(self, ex, image, cands, **kwargs):
                return {"qid": f"eb_{ex['index']}", "correct": False, "logp_gold": 0,
                        "scene": "test", "task": "test", "category": "test"}

            def close(self):
                pass

        argv = ["eval_endobench.py", "--mode", "baseline", "--benchmark-file", str(self.source),
                "--qid-file", str(selected), "--out-dir", str(self.root / "fake_results")]
        with patch.object(sys, "argv", argv), patch.object(module, "EndoBenchEvaluator", FakeEvaluator), \
                patch("datasets.load_dataset", side_effect=AssertionError("不允许联网下载题库")):
            module.main()
        result = list(portable_data.read_jsonl(self.root / "fake_results/baseline_samples.jsonl"))
        self.assertEqual([row["qid"] for row in result], ["eb_42"])


if __name__ == "__main__":
    (HERE / "assets").mkdir(exist_ok=True)
    unittest.main()
