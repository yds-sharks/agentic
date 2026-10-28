#!/usr/bin/env python3
"""从本机已缓存的数据导出可跨机评测的题库和证据图片；不启动模型。"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sqlite3
import tarfile
from pathlib import Path

from datasets import Dataset
from PIL import Image


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def image_refs(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "image_path" and isinstance(value, str) and value:
                yield value
            elif isinstance(value, (list, dict)):
                yield from image_refs(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from image_refs(value)


def database_refs(db):
    refs = set()
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for (table,) in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({quoted})")]
            for col in cols:
                if col == "image_path":
                    refs.update(row[0] for row in conn.execute(
                        f'SELECT DISTINCT "image_path" FROM {quoted} WHERE "image_path" IS NOT NULL'
                    ) if row[0])
                elif col == "images":
                    for (raw,) in conn.execute(f'SELECT "images" FROM {quoted}'):
                        if raw:
                            refs.update(image_refs(json.loads(raw)))
    return refs


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrow", required=True)
    ap.add_argument("--samples-db", required=True)
    ap.add_argument("--cand-cache", required=True)
    ap.add_argument("--translate-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    ds = Dataset.from_file(args.arrow)
    refs = database_refs(args.samples_db)
    cache_qids = set()
    cache_refs = set()
    for row in jsonl(args.cand_cache):
        cache_qids.add(row["qid"])
        cache_refs.update(image_refs(row))
    refs.update(cache_refs)
    missing = sorted(p for p in refs if not Path(p).is_file())
    size = sum(Path(p).stat().st_size for p in refs if Path(p).is_file())
    summary = {
        "questions": len(ds), "columns": ds.column_names,
        "cache_questions": len(cache_qids), "cache_image_refs": len(cache_refs),
        "corpus_image_refs": len(refs), "corpus_image_bytes": size,
        "missing_corpus_images": len(missing), "missing_examples": missing[:5],
        "query_image_examples": [str(ds[i].get("image", ""))[:64] for i in range(3)],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.inspect:
        return
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 张证据图片，不能发布不完整的评测包")
    if len(ds) != 6832:
        raise ValueError(f"题库数量异常: {len(ds)}")
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"输出目录非空，避免覆盖既有资产: {out}")
    benchmark = out / "payload/assets/benchmark"
    benchmark.mkdir(parents=True)
    query_dir = benchmark / "query_images"
    query_dir.mkdir()
    query_count = 0
    query_ids = set()
    with (benchmark / "endobench_test.jsonl").open("w", encoding="utf-8") as stream:
        for row in ds:
            idx = int(row["index"])
            if idx in query_ids:
                raise ValueError(f"重复 index: {idx}")
            query_ids.add(idx)
            raw = row.pop("image", "")
            row.pop("image_path", None)
            row["query_image_path"] = ""
            if raw:
                # 保留原始图像字节，不重压缩、不改变视觉评测输入。
                if str(raw).isdigit():
                    raw = ds[int(raw)]["image"]
                path = query_dir / f"eb_{idx}.jpg"
                path.write_bytes(base64.b64decode(raw, validate=True))
                with Image.open(path) as image:
                    image.verify()
                row["query_image_path"] = f"query_images/eb_{idx}.jpg"
                query_count += 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    mapping = {}
    members = {}
    for original in sorted(refs):
        path = Path(original)
        with Image.open(path) as image:
            image.verify()
        name = hashlib.sha256(original.encode()).hexdigest() + path.suffix.lower()
        relative = f"assets/corpus_images/{name}"
        mapping[original] = relative
        members[relative] = path
    (benchmark / "image_path_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    translations = list(jsonl(args.translate_cache))
    with (benchmark / "endobench_translated_queries.jsonl").open("w", encoding="utf-8") as stream:
        for row in translations:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (benchmark / "cand_cache_v2.jsonl").write_bytes(Path(args.cand_cache).read_bytes())
    manifest = {
        "format_version": 1, "source": "Saint-lsy/EndoBench", "split": "test",
        "source_revision": Path(args.arrow).parent.name,
        "questions": len(ds), "query_images": query_count,
        "candidate_cache_questions": len(cache_qids), "corpus_images": len(mapping),
        "translation_rows": len(translations),
        "image_paths": "题库路径相对 benchmark；映射目标相对 endobench_eval_pkg",
        "attribution": "EndoBench: arXiv:2505.23601；原始数据及语料图片权利归各来源所有，使用须遵循上游条件。",
        "files": {p.name: sha256(p) for p in benchmark.iterdir() if p.is_file()},
    }
    (benchmark / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    archive = out / "endobench_data_v1.tar.gz"
    with tarfile.open(archive, "w:gz", dereference=True) as tar:
        tar.add(benchmark, arcname="assets/benchmark")
    corpus = out / "evidence_images_v1.tar.gz"
    with tarfile.open(corpus, "w:gz", compresslevel=1, dereference=True) as tar:
        for number, (relative, path) in enumerate(members.items(), 1):
            tar.add(path, arcname=relative)
            if number % 10000 == 0:
                print(f"已打包 {number}/{len(members)} 张证据图", flush=True)
    # 每片 512 MiB，避免超过 GitHub Release 单文件大小限制。
    parts = []
    with corpus.open("rb") as stream:
        number = 0
        while block := stream.read(512 * 1024 * 1024):
            part = out / f"evidence_images_v1.tar.gz.part{number:02d}"
            part.write_bytes(block)
            parts.append(part)
            number += 1
    publish = [archive] + parts
    checksum_file = out / "EVAL_DATA_SHA256SUMS.txt"
    checksum_file.write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in publish), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"发布资产目录: {out}", flush=True)


if __name__ == "__main__":
    main()
