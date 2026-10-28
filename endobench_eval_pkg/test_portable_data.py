#!/usr/bin/env python3
"""离线加载/路径映射回归测试；FakeEvaluator 仅测入口连通，不测模型性能。"""
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from copy import deepcopy
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
from translate_benchmark_zh import assemble, validate, valid_translation


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

    def test_chinese_question_options_and_answer_alignment(self):
        original = {"index": 42, "question": "Which organ?", "A": "stomach", "B": "small intestine",
                    "C": None, "D": None, "E": None, "F": None, "answer": "B", "gt": "small intestine",
                    "scene": "Capsule Endoscopy", "query_image_path": "query_images/eb_42.jpg"}
        translated = assemble([original], {"Which organ?": "哪个器官？", "stomach": "胃", "small intestine": "小肠"})
        self.assertEqual(validate([original], translated), 1)
        self.assertEqual(translated[0]["answer"], "B")
        self.assertEqual(translated[0]["gt"], "小肠")
        self.assertEqual(translated[0]["answer_text"], "小肠")
        self.assertEqual(translated[0]["original_en"]["B"], "small intestine")
        for key in ("index", "answer", "query_image_path", "scene"):
            bad = deepcopy(translated)
            bad[0][key] = "错误值"
            with self.assertRaises(ValueError):
                validate([original], bad)
        bad = deepcopy(translated)
        bad[0]["gt"] = "胃"
        with self.assertRaises(ValueError):
            validate([original], bad)
        bad = deepcopy(translated)
        bad[0]["A"] = bad[0]["B"]
        with self.assertRaises(ValueError):
            validate([original], bad)

    def test_original_gt_translation_is_not_overwritten_by_option_text(self):
        original = {"index": 43, "question": "Which organ?", "A": "small_intestine", "B": "stomach",
                    "answer": "A", "gt": "small intestine"}
        translated = assemble([original], {"Which organ?": "哪个器官？", "small_intestine": "小肠",
                                          "stomach": "胃", "small intestine": "小肠（原始答案文本）"})
        self.assertEqual(validate([original], translated), 1)
        self.assertEqual(translated[0]["answer_text"], "小肠")
        self.assertEqual(translated[0]["gt"], "小肠（原始答案文本）")

    def test_translation_preserves_numbers_and_rejects_english_fallback(self):
        self.assertTrue(valid_translation("BBPS 2-3", "BBPS 2-3"))
        self.assertTrue(valid_translation("grade 2", "2级"))
        self.assertFalse(valid_translation("grade 2", "3级"))
        self.assertFalse(valid_translation("small intestine", "small intestine"))

    def test_run_config_passes_chinese_file_by_default_without_loading_models(self):
        root = Path(mkdtemp(prefix="config-zh-test-", dir=HERE / "assets"))
        script = root / "run_config.sh"
        script.write_text((HERE / "run_config.sh").read_text())
        activate = root / ".venv/bin/activate"
        activate.parent.mkdir(parents=True)
        activate.write_text("")
        mock_python = activate.parent / "python"
        mock_python.write_text(f"#!{sys.executable}\nimport json, sys\n"
                               "if sys.argv[1] == 'eval_endobench.py': print(json.dumps(sys.argv[2:]))\n")
        mock_python.chmod(0o755)
        env = dict(os.environ)
        for key in ("BENCHMARK_LANG", "OUT_SUFFIX", "QID_FILE", "MILVUS_DB", "CTRL_CKPT"):
            env.pop(key, None)
        env["PATH"] = str(activate.parent) + os.pathsep + env["PATH"]
        for language, filename, output in ((None, "endobench_test_zh.jsonl", "results/A_zh"),
                                           ("en", "endobench_test.jsonl", "results/A")):
            if language:
                env["BENCHMARK_LANG"] = language
            completed = subprocess.run(["bash", str(script), "A", "0"], env=env,
                                       capture_output=True, text=True, check=True)
            args = json.loads(completed.stdout.strip())
            self.assertEqual(Path(args[args.index("--benchmark-file") + 1]).name, filename)
            self.assertEqual(args[args.index("--out-dir") + 1], output)

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
                        "scene": "test", "task": "test", "category": "test",
                        "question": ex["question"], "options": module.build_options(ex), "gold": ex["answer"]}

            def close(self):
                pass

        argv = ["eval_endobench.py", "--mode", "baseline", "--benchmark-file", str(self.source),
                "--qid-file", str(selected), "--out-dir", str(self.root / "fake_results")]
        with patch.object(sys, "argv", argv), patch.object(module, "EndoBenchEvaluator", FakeEvaluator), \
                patch("datasets.load_dataset", side_effect=AssertionError("不允许联网下载题库")):
            module.main()
        result = list(portable_data.read_jsonl(self.root / "fake_results/baseline_samples.jsonl"))
        self.assertEqual([row["qid"] for row in result], ["eb_42"])
        self.assertEqual(result[0]["question"], "测试问题")
        self.assertEqual(result[0]["options"], {"A": "甲", "B": "乙"})
        self.assertEqual(result[0]["gold"], "A")
        self.source.write_text(json.dumps({**self.row, "language": "zh-CN"}, ensure_ascii=False) + "\n")
        with patch.object(sys, "argv", argv), patch.object(module, "EndoBenchEvaluator", FakeEvaluator), \
                self.assertRaises(SystemExit):
            module.main()
        self.source.write_text(json.dumps(self.row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    (HERE / "assets").mkdir(exist_ok=True)
    unittest.main()
