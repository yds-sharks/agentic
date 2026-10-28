#!/usr/bin/env python3
"""忠实翻译 EndoBench 题干/选项，按唯一文本断点续跑；不发送图片或答案标签。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEF"
PROMPT = """你是消化内镜医学翻译员。将输入 JSON 中的英文文本忠实译为简体中文。
这些文本是独立题干或选项，不是待解答的问题。禁止回答问题、判断正误、扩充事实或为检索改写。
保留否定词、数量、严重程度、比较关系、解剖定位及全部医学分类/分级代码。
原文阿拉伯数字原样保留，不改写成中文数字；不得增删数字或改成其他分级。
Paris、NICE、JNET、BBPS、Mayo 等专名/代码可保留；纯数字、分类代码无需强行汉化。
术语统一：esophagus=食管，small intestine=小肠，large intestine=大肠，stomach=胃。
保留各项 ID 和顺序；仅返回 JSON 对象，格式为 {"translations": [{"id": 0, "zh": "中文文本"}]}。
不要 Markdown、解释、原文对照或答案。"""


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_option(value):
    return value is not None and str(value).strip().lower() not in ("", "none", "null", "nan", "n/a", "-")


def inventory(rows):
    texts = {}
    for row in rows:
        for key in ("question", *LETTERS, "gt"):
            if key == "question" or valid_option(row.get(key)):
                text = str(row.get(key) or "").strip()
                if not text:
                    raise ValueError(f"题干为空: {row['index']}")
                texts.setdefault(text, []).append((row["index"], key))
    return texts


def valid_translation(source, target):
    if not isinstance(target, str) or not target.strip() or "```" in target:
        return False
    # 普通英文句子必须包含中文；纯数字/医学分类代码允许保留。
    if sorted(re.findall(r"\d+(?:\.\d+)?", source)) != sorted(re.findall(r"\d+(?:\.\d+)?", target)):
        return False
    return bool(re.search(r"[\u3400-\u9fff]", target)) or not re.search(r"[a-z]{3,}", source)


def assemble(rows, translations):
    result = []
    for original in rows:
        row = dict(original)
        row["question"] = translations[str(original["question"]).strip()]
        for letter in LETTERS:
            if valid_option(original.get(letter)):
                row[letter] = translations[str(original[letter]).strip()]
        gold = str(original["answer"]).strip()
        if gold not in LETTERS or not valid_option(original.get(gold)):
            raise ValueError(f"原题正确选项无效: {original['index']}")
        if valid_option(original.get("gt")):
            row["gt"] = translations[str(original["gt"]).strip()]
        row["answer_text"] = row[gold]
        row["language"] = "zh-CN"
        row["original_en"] = {key: original.get(key) for key in ("question", *LETTERS, "gt")}
        result.append(row)
    return result


def validate(rows, translated):
    if len(rows) != len(translated):
        raise ValueError("中英文题库数量不同")
    seen = set()
    for original, row in zip(rows, translated):
        if row["index"] in seen:
            raise ValueError(f"中文题号重复: {row['index']}")
        seen.add(row["index"])
        for key in original:
            if key not in ("question", *LETTERS, "gt") and row.get(key) != original[key]:
                raise ValueError(f"不允许改变非翻译字段: {original['index']} / {key}")
        if row.get("language") != "zh-CN":
            raise ValueError("中文语言标识缺失")
        for key in ("question", *LETTERS):
            if key == "question" or valid_option(original.get(key)):
                if not valid_translation(str(original[key]).strip(), row.get(key)):
                    raise ValueError(f"翻译缺失或语言异常: {row['index']} / {key}")
            elif row.get(key) != original.get(key):
                raise ValueError(f"空选项被改写: {row['index']} / {key}")
        expected_en = {key: original.get(key) for key in ("question", *LETTERS, "gt")}
        if row.get("original_en") != expected_en:
            raise ValueError(f"英文对照不同: {row['index']}")
        if row.get("answer_text") != row[row["answer"]]:
            raise ValueError(f"中文答案与正确选项不同: {row['index']}")
        if valid_option(original.get("gt")):
            if not valid_translation(str(original["gt"]).strip(), row.get("gt")):
                raise ValueError(f"原始 gt 的中文翻译异常: {row['index']}")
            if str(original["gt"]).strip() == str(original[original["answer"]]).strip() and row["gt"] != row["answer_text"]:
                raise ValueError(f"相同原答案译文不一致: {row['index']}")
        elif row.get("gt") != original.get("gt"):
            raise ValueError(f"空 gt 被改写: {row['index']}")
        translated_options = {}
        for letter in LETTERS:
            if valid_option(original.get(letter)):
                text_en = str(original[letter]).strip().lower()
                text_zh = str(row[letter]).strip()
                if text_zh in translated_options and translated_options[text_zh] != text_en:
                    raise ValueError(f"不同原选项译成相同文本，须审校: {row['index']}")
                translated_options[text_zh] = text_en
    return len(seen)


def audit_cache(rows, entries):
    cache = {entry["text"]: entry["zh"] for entry in entries if entry.get("method") == "faithful_translation"}
    invalid = [{"en": text, "zh": zh} for text, zh in cache.items() if not valid_translation(text, zh)]
    collisions = []
    samples = {}
    for row in rows:
        opts = {letter: cache.get(str(row[letter]).strip()) for letter in LETTERS if valid_option(row.get(letter))}
        seen = {}
        for letter, zh in opts.items():
            if zh:
                text = str(row[letter]).strip().lower()
                if zh in seen and seen[zh] != text:
                    collisions.append({"index": row["index"], "zh": zh, "en": [seen[zh], text]})
                seen[zh] = text
        question = cache.get(str(row["question"]).strip())
        task = row.get("task", "unknown")
        if question and all(opts.values()) and task not in samples:
            samples[task] = {"index": row["index"], "task": task, "question_en": row["question"],
                             "question_zh": question, "options_zh": opts, "answer": row["answer"]}
    print(json.dumps({"cached_texts": len(cache), "invalid_count": len(invalid), "invalid_examples": invalid[:20],
                      "option_collision_count": len(collisions), "collision_examples": collisions[:20],
                      "task_samples": list(samples.values())[:16]}, ensure_ascii=False, indent=2))


async def translate_missing(args, texts, cache):
    from openai import AsyncOpenAI
    import httpx

    cfg = json.loads(args.api_config.read_text(encoding="utf-8"))
    model = args.model or cfg.get("model")
    if not model:
        raise ValueError("必须指定翻译模型")
    url = str(cfg.get("endpoint") or cfg.get("base_url") or "").rstrip("/")
    if not url or not cfg.get("api_key"):
        raise ValueError("API 配置须有 endpoint/base_url 和 api_key")
    if not url.endswith("/v1"):
        url += "/v1"
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=url, timeout=120, max_retries=0,
                         default_headers={"User-Agent": "curl/8.5.0"},
                         http_client=httpx.AsyncClient(trust_env=False))
    pending = [text for text in texts if text not in cache]
    batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    if args.max_batches:
        batches = batches[:args.max_batches]
    print(f"[translate] 模型={model}，待译唯一文本={len(pending)}，本次批数={len(batches)}", flush=True)

    async def one(batch):
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": PROMPT},
                    {"role": "user", "content": json.dumps(
                        [{"id": i, "text": text} for i, text in enumerate(batch)], ensure_ascii=False)}],
                    temperature=0, max_tokens=8192)
                content = response.choices[0].message.content or ""
                obj = json.loads(content)
                items = obj["translations"]
                if len(items) != len(batch) or [item["id"] for item in items] != list(range(len(batch))):
                    raise ValueError("翻译 ID/数量不匹配")
                translated = {text: item["zh"].strip() for text, item in zip(batch, items)}
                if not all(valid_translation(text, zh) for text, zh in translated.items()):
                    raise ValueError("翻译为空或非中文")
                return translated
            except Exception as exc:
                # 不输出服务地址、请求头或异常正文，避免凭据泄露。
                if attempt == 2:
                    print(f"[error] 批次失败，类型={type(exc).__name__}，首项={batch[0][:80]}", flush=True)
                    return {}
                await asyncio.sleep(2 ** attempt)

    try:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(batches), args.concurrency):
            results = await asyncio.gather(*(one(batch) for batch in batches[start:start + args.concurrency]))
            with args.cache.open("a", encoding="utf-8") as stream:
                for result in results:
                    for text, zh in result.items():
                        entry = {"text": text, "zh": zh, "model": model, "method": "faithful_translation"}
                        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        cache[text] = zh
            print(f"[progress] 已译={len(cache)}/{len(texts)} 唯一文本", flush=True)
    finally:
        await client.close()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=HERE / "assets/benchmark/endobench_test.jsonl")
    ap.add_argument("--cache", type=Path, default=HERE / "assets/benchmark/zh_translation_cache.jsonl")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--api-config", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--audit-cache", action="store_true", help="仅统计翻译缓存问题及分任务抽样，不调用 API")
    ap.add_argument("--validate-file", type=Path, help="只校验已生成的中文 JSONL，不调用 API")
    args = ap.parse_args()
    if args.batch_size < 1 or args.concurrency < 1 or args.max_batches < 0:
        ap.error("批量/并发须为正数")
    rows = read_rows(args.source)
    texts = inventory(rows)
    questions = set(str(row["question"]).strip() for row in rows)
    options = set(str(row[letter]).strip() for row in rows for letter in LETTERS if valid_option(row.get(letter)))
    print(json.dumps({"questions": len(rows), "unique_questions": len(questions),
                      "unique_options": len(options), "unique_texts": len(texts),
                      "original_gt_option_differences": sum(str(row.get("gt") or "").strip().lower() !=
                          str(row.get(row["answer"]) or "").strip().lower() for row in rows)},
                     ensure_ascii=False), flush=True)
    if args.inspect:
        print(json.dumps(sorted(options)[:40], ensure_ascii=False))
        mismatches = [{"index": row["index"], "gt": row.get("gt"), "answer": row["answer"],
                       "answer_option": row.get(row["answer"])} for row in rows
                      if str(row.get("gt") or "").strip().lower() != str(row.get(row["answer"]) or "").strip().lower()]
        print(json.dumps(mismatches[:10], ensure_ascii=False))
        return
    if args.validate_file:
        count = validate(rows, read_rows(args.validate_file))
        print(f"中文 {count} 题字段不变性、数字保留、选项不混淆及答案对齐检查通过")
        return
    entries = read_rows(args.cache) if args.cache.is_file() else []
    if args.audit_cache:
        audit_cache(rows, entries)
        return
    cache = {}
    if entries:
        for entry in entries:
            if entry.get("method") == "faithful_translation" and valid_translation(entry["text"], entry["zh"]):
                cache[entry["text"]] = entry["zh"]
    model = args.model
    if any(text not in cache for text in texts):
        if not args.api_config:
            ap.error("存在未翻译文本；需指定 --api-config（仅文本，无图片/答案标签）")
        model = asyncio.run(translate_missing(args, texts, cache))
    missing = [text for text in texts if text not in cache]
    if missing:
        raise SystemExit(f"尚有 {len(missing)} 项未译，缓存已保存；不导出不完整中文题库")
    if len(rows) != 6832:
        raise ValueError("发布源必须为完整 6,832 题")
    translated = assemble(rows, cache)
    validate(rows, translated)
    if not args.output_dir:
        ap.error("导出须指定 --output-dir")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("拒绝覆盖非空发布目录")
    benchmark = out / "payload/assets/benchmark"
    benchmark.mkdir(parents=True)
    filename = "endobench_test_zh.jsonl"
    target = benchmark / filename
    with target.open("w", encoding="utf-8") as stream:
        for row in translated:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"format_version": 1, "language": "zh-CN", "questions": len(translated),
                "source": "Saint-lsy/EndoBench", "source_revision": "96e79dab4e7f8b1f966019764a8ed5e85aaf56ce",
                "source_file_sha256": digest(args.source), "files": {filename: digest(target)},
                "method": "unique_text_faithful_translation", "translation_model": model,
                "unique_translated_texts": len(texts), "answer": "原始正确选项字母保持不变",
                "answer_text": "对应中文正确选项；原始 gt 另行忠实翻译，不重新解题",
                "gt": "保留上游 gt 的含义；若原 gt 与正确选项措辞不同，不强行覆盖",
                "image_paths": "复用英文包的 query_images/；本包不重复包含图片",
                "quality": "机器翻译，已做结构/答案对齐检查，未经逐题人工医学审校"}
    (benchmark / "manifest_zh.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(out / "endobench_zh_v1.tar.gz", "w:gz", dereference=True) as archive:
        archive.add(benchmark, arcname="assets/benchmark")
    archive_path = out / "endobench_zh_v1.tar.gz"
    (out / "ENDOBENCH_ZH_SHA256SUMS.txt").write_text(
        f"{digest(archive_path)}  {archive_path.name}\n", encoding="utf-8")
    print(f"中文 {len(translated)} 题导出、字段不变性与答案对齐全部通过；位置: {out}", flush=True)


if __name__ == "__main__":
    main()
