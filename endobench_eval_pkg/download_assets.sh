#!/bin/bash
# 从 GitHub Release 下载运行资产并重组（私有仓库需先 gh auth login）
set -euo pipefail
PKGROOT="$(cd "$(dirname "$0")" && pwd)"
REL=eval-assets-v1
REPO=yds-sharks/agentic
DL="$PKGROOT/assets/dl"
MODE=${1:-all}
if [[ "$MODE" != all && "$MODE" != --benchmark-only && "$MODE" != --zh-only ]]; then
  echo "用法: bash download_assets.sh [--benchmark-only|--zh-only]" >&2
  exit 2
fi
mkdir -p "$DL" "$PKGROOT/models" "$PKGROOT/assets"
# 已有下载保留，重新部署只补缺少的文件。
download() {
  if [[ ! -s "$DL/$1" ]]; then
    gh release download "$REL" --repo "$REPO" --dir "$DL" --pattern "$1"
  fi
}
download_zh() {
  download ENDOBENCH_ZH_SHA256SUMS.txt
  while read -r hash name; do
    [[ -n "$hash" ]] || continue
    [[ "$name" == endobench_zh_v1.tar.gz ]] || { echo "中文校验清单路径非法" >&2; exit 1; }
    download "$name"
  done < "$DL/ENDOBENCH_ZH_SHA256SUMS.txt"
  (cd "$DL" && sha256sum -c ENDOBENCH_ZH_SHA256SUMS.txt)
  tar xzf "$DL/endobench_zh_v1.tar.gz" -C "$PKGROOT"
}
if [[ "$MODE" == --zh-only ]]; then
  if [[ ! -s "$PKGROOT/assets/benchmark/endobench_test.jsonl" ]]; then
    echo "请先运行 bash download_assets.sh --benchmark-only 获取基础题库及图片" >&2
    exit 1
  fi
  download_zh
  python3 "$PKGROOT/check_eval_data.py" --language both
  echo "[dl] 中文题干、选项和答案就位；图片及模型未重复下载"
  exit 0
fi
echo "[dl] 下载离线题库、查询图和证据图 $(date +%H:%M:%S)"
download EVAL_DATA_SHA256SUMS.txt
while read -r hash name; do
  [[ -n "$hash" ]] || continue
  [[ "$name" != */* && "$name" != .* ]] || { echo "校验清单路径非法" >&2; exit 1; }
  download "$name"
done < "$DL/EVAL_DATA_SHA256SUMS.txt"
(cd "$DL" && sha256sum -c EVAL_DATA_SHA256SUMS.txt)
tar xzf "$DL/endobench_data_v1.tar.gz" -C "$PKGROOT"
cat "$DL"/evidence_images_v1.tar.gz.part* | tar xzf - -C "$PKGROOT"
download_zh
# 保留旧脚本使用的缓存位置；不覆盖已有评测缓存。
for name in cand_cache_v2.jsonl endobench_translated_queries.jsonl; do
  if [[ ! -f "$PKGROOT/assets/$name" ]]; then
    cp "$PKGROOT/assets/benchmark/$name" "$PKGROOT/assets/$name"
  fi
done
python3 "$PKGROOT/check_eval_data.py" --language both
if [[ "$MODE" == --benchmark-only ]]; then
  echo "[dl] 评测数据就位；未下载模型或启动 GPU 评测"
  exit 0
fi
download_core() {
  download "$1"
  awk -v name="$1" '$2 == name' "$PKGROOT/core_assets.sha256" \
    | (cd "$DL" && sha256sum -c -)
}
echo "[dl] reassemble u50 checkpoint $(date +%H:%M:%S)"
if [[ ! -f "$PKGROOT/models/ckpt_grpo_v4_u50/config.json" ]]; then
  for n in 00 01 02 03 04; do download_core "u50.tar.part_$n"; done
  cat "$DL"/u50.tar.part_* | tar xf - -C "$PKGROOT/models"
fi
echo "[dl] reassemble milvus db $(date +%H:%M:%S)"
if [[ ! -s "$PKGROOT/assets/multimodal_vector_indexes.db" ]]; then
  for n in 00 01 02; do download_core "milvus.db.part_$n"; done
  cat "$DL"/milvus.db.part_* > "$PKGROOT/assets/multimodal_vector_indexes.db"
fi
printf '%s  %s\n' e601a70151e24f7be2bb94d305043638a4b43a19048fc7b07538aa009ca49c29 \
  "$PKGROOT/assets/multimodal_vector_indexes.db" | sha256sum -c -
if [[ ! -s "$PKGROOT/assets/multimodal_samples.db" ]]; then
  download_core multimodal_samples.db
  cp "$DL/multimodal_samples.db" "$PKGROOT/assets/multimodal_samples.db"
fi
printf '%s  %s\n' 5ddc87d0e58f7880adf2dac84ae1bd48716e62fe6141d6c470fa9530349d0de3 \
  "$PKGROOT/assets/multimodal_samples.db" | sha256sum -c -
if [[ ! -f "$PKGROOT/weight_module/checkpoints/.eval_assets_ready" ]]; then
  download_core weight_ckpts.tar
  mkdir -p "$PKGROOT/weight_module/checkpoints"
  tar xf "$DL/weight_ckpts.tar" -C "$PKGROOT/weight_module/checkpoints"
  touch "$PKGROOT/weight_module/checkpoints/.eval_assets_ready"
fi
echo "[dl] ASSETS READY $(date +%H:%M:%S)"
