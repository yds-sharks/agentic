"""离线题库加载与跨机图片映射；保持候选 ID 和原始路径元数据不变。"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PKGROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = PKGROOT / "assets/benchmark/endobench_test.jsonl"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_local_benchmark(path):
    path = Path(path).resolve()
    rows = list(read_jsonl(path))
    seen = set()
    for row in rows:
        idx = int(row["index"])
        if idx in seen:
            raise ValueError(f"题库存在重复 index: {idx}")
        seen.add(idx)
        image = str(row.get("query_image_path") or "")
        if image:
            local = (path.parent / image).resolve()
            if not local.is_file():
                raise FileNotFoundError(f"缺少查询图片: {local}")
            row["query_image_path"] = str(local)
    return rows


@lru_cache(maxsize=1)
def image_path_map():
    path = Path(os.environ.get(
        "EVAL_IMAGE_PATH_MAP", PKGROOT / "assets/benchmark/image_path_map.json"
    ))
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def resolve_image_path(original):
    if not original:
        return ""
    mapping = image_path_map()
    if original in mapping:
        path = (PKGROOT / mapping[original]).resolve()
    else:
        path = Path(original)
    if not path.is_file():
        raise FileNotFoundError(
            f"图片不可读: {original}；请运行 bash download_assets.sh --benchmark-only 补齐评测图片"
        )
    return str(path)
