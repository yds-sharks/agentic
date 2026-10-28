#!/usr/bin/env python3
"""使用已生成的 Release 资产测试真实下载脚本的解包链路；只做 CPU 验证。"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify_remote(release, checksums_name="EVAL_DATA_SHA256SUMS.txt"):
    metadata = json.loads(subprocess.check_output([
        "gh", "api", "repos/yds-sharks/agentic/releases/tags/eval-assets-v1"
    ], text=True))
    assets = {item["name"]: item for item in metadata["assets"]}
    checksums = release / checksums_name
    expected = dict((line.split()[1], line.split()[0]) for line in
                    checksums.read_text().splitlines() if line.strip())
    expected[checksums.name] = hashlib.sha256(checksums.read_bytes()).hexdigest()
    for name, digest in expected.items():
        asset = assets.get(name)
        if not asset or asset["state"] != "uploaded":
            raise ValueError(f"远端未完整上传: {name}")
        if asset["size"] != (release / name).stat().st_size:
            raise ValueError(f"远端文件大小不一致: {name}")
        if asset.get("digest") != "sha256:" + digest:
            raise ValueError(f"远端服务端 SHA256 不一致: {name}")
    print(f"远端 {len(expected)} 个资产大小、uploaded 状态、服务端 SHA256 全部通过（{checksums_name}）")
    return len(expected)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-dir", required=True, type=Path)
    ap.add_argument("--source-arrow", type=Path)
    ap.add_argument("--zh-release-dir", type=Path, help="中文增量 Release 目录")
    ap.add_argument("--zh-only", action="store_true", help="仅真实解包中文增量，要求基础数据已就位")
    ap.add_argument("--remote-only", action="store_true", help="仅验证远端 Release 文件大小和服务端 SHA256")
    args = ap.parse_args()
    release = args.release_dir.resolve()
    if args.remote_only:
        count = verify_remote(release)
        if args.zh_release_dir:
            count += verify_remote(args.zh_release_dir.resolve(), "ENDOBENCH_ZH_SHA256SUMS.txt")
        print(f"合计 {count} 个远端资产验证通过")
        return
    if not args.zh_release_dir:
        ap.error("新版下载链同时安装完整中文题库；须指定 --zh-release-dir")
    dl = HERE / "assets/dl"
    dl.mkdir(parents=True, exist_ok=True)
    sources = []
    manifests = [(args.zh_release_dir.resolve(), "ENDOBENCH_ZH_SHA256SUMS.txt")]
    if not args.zh_only:
        manifests.append((release, "EVAL_DATA_SHA256SUMS.txt"))
    for folder, checksums_name in manifests:
        sources.append(folder / checksums_name)
        sources += [folder / line.split()[1] for line in
                    (folder / checksums_name).read_text().splitlines() if line.strip()]
    for source in sources:
        target = dl / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.exists():
            target.symlink_to(source)
        elif target.resolve() != source:
            raise FileExistsError(f"下载缓存已有不同文件，拒绝覆盖: {target}")
    mode = "--zh-only" if args.zh_only else "--benchmark-only"
    subprocess.run(["bash", str(HERE / "download_assets.sh"), mode], check=True)
    if args.source_arrow:
        from datasets import Dataset
        ds = Dataset.from_file(str(args.source_arrow))
        benchmark = HERE / "assets/benchmark"
        rows = [json.loads(line) for line in
                (benchmark / "endobench_test.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(ds):
            raise ValueError("导出前后题库数量不同")
        for original, row in zip(ds, rows):
            for key in ds.column_names:
                if key not in ("image", "image_path") and original[key] != row[key]:
                    raise ValueError(f"导出改变了原题字段: {row['index']} / {key}")
            raw = original.get("image")
            if raw:
                if str(raw).isdigit():
                    raw = ds[int(raw)]["image"]
                if (benchmark / row["query_image_path"]).read_bytes() != base64.b64decode(raw):
                    raise ValueError(f"导出改变了查询图片字节: {row['index']}")
        print(f"原始 Arrow 对账通过：{len(rows)} 题字段和查询图片字节全部一致")
    print("真实解包、SHA256、全量数据检查均通过；未运行模型或 GPU 评测")


if __name__ == "__main__":
    main()
