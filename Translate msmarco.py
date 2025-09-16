import argparse
import gzip
import json
import os
import random
import re
import sys
import time
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
from tqdm import tqdm



DEFAULT_INPUT_FILE = "msmarco-passage-aug/train.jsonl.gz"
DEFAULT_SAMPLE_FILE = "sampled_subset.jsonl"
DEFAULT_OUT_PARQUET = "tevatron_msmarco_ru.parquet"
DEFAULT_OUT_JSONL = "tevatron_msmarco_ru.jsonl"
DEFAULT_N_SAMPLES = 10_000
DEFAULT_MAX_HNEGS = 5
DEFAULT_BATCH_EXAMPLES = 10           
DEFAULT_CHECKPOINT_INTERVAL = 200     
DEFAULT_COOLDOWN_SECONDS = 3600      
DEFAULT_SEED = 42


MAX_TOTAL_CHARS_PER_CALL = 9500       
MAX_SINGLE_TEXT_CHARS = 9000          


MAX_CALLS_PER_SEC = 20
MAX_CHARS_PER_HOUR = 1_000_000
REQUEST_MIN_INTERVAL = 0.1            


_last_call_ts = 0.0
_window_start = time.time()
_chars_this_hour = 0


# Yandex Translate 

def translate_batch_yandex(
    texts: List[str],
    api_key: str,
    folder_id: str,
    target_lang: str = "ru",
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    url: str = "https://translate.api.cloud.yandex.net/translate/v2/translate",
) -> List[str]:
    """
    Translate a list of texts with Yandex; preserves order; retries on errors.
    """
    global _last_call_ts, _window_start, _chars_this_hour

    now = time.time()
    elapsed = now - _window_start
    if elapsed >= 3600:
        _window_start = now
        _chars_this_hour = 0

    batch_chars = sum(len(t) for t in texts)
    if _chars_this_hour + batch_chars > MAX_CHARS_PER_HOUR:
        sleep_s = max(0, 3600 - elapsed)
        print(f"⏸️ Approaching 1M chars/hour. Sleeping {sleep_s:.0f}s to reset window...")
        time.sleep(sleep_s)
        _window_start = time.time()
        _chars_this_hour = 0

    since_last = time.time() - _last_call_ts
    if since_last < REQUEST_MIN_INTERVAL:
        time.sleep(REQUEST_MIN_INTERVAL - since_last)

    headers = {"Authorization": f"Api-Key {api_key}"}
    body = {"targetLanguageCode": target_lang, "texts": texts, "folderId": folder_id}

    while True:
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=60)
            _last_call_ts = time.time()

            if resp.status_code == 429:
                print("⚠️  Yandex quota exceeded (HTTP 429). Cooling down...")
                time.sleep(cooldown_seconds)
                continue

            resp.raise_for_status()
            data = resp.json()
            out = [t["text"] for t in data["translations"]]
            if len(out) != len(texts):
                raise RuntimeError(f"Mismatch in translation count: sent {len(texts)} got {len(out)}")

            _chars_this_hour += batch_chars
            return out

        except requests.exceptions.HTTPError as e:
            print("❌ HTTP error:", e)
            print("➡️ Response:", getattr(resp, "text", "")[:500])
            time.sleep(10)
        except Exception as e:
            print("❌ General error:", e)
            time.sleep(10)



def _split_long_text(text: str, max_len: int = MAX_SINGLE_TEXT_CHARS) -> List[str]:
    """
    Split a long text into <= max_len chunks on sentence boundaries or whitespace.
    """
    if len(text) <= max_len:
        return [text]

    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks, buf = [], ""
    for s in sentences:
        if not s:
            continue
        candidate = (buf + " " + s).strip() if buf else s
        if len(candidate) <= max_len:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(s) > max_len:
                # hard split very long sentence
                for i in range(0, len(s), max_len):
                    chunks.append(s[i:i + max_len])
                buf = ""
            else:
                buf = s
    if buf:
        chunks.append(buf)
    return chunks


def _explode_texts(texts: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    For a list of texts, split any text > MAX_SINGLE_TEXT_CHARS.
    Returns:
      flat_chunks: list[str] with all chunks in order
      rebuild_map: list of (start_idx_in_flat, n_chunks) for each original text
    """
    flat_chunks = []
    rebuild_map = []
    for t in texts:
        parts = _split_long_text(t, MAX_SINGLE_TEXT_CHARS)
        rebuild_map.append((len(flat_chunks), len(parts)))
        flat_chunks.extend(parts)
    return flat_chunks, rebuild_map


def _pack_by_total_chars(texts: List[str], max_total_chars: int = MAX_TOTAL_CHARS_PER_CALL) -> List[Tuple[int, int]]:
    """
    Greedy pack texts into sub-batches so each sub-batch has total length <= max_total_chars.
    Returns list of (start, end) slices into the 'texts' list.
    """
    slices = []
    start = 0
    cur_len = 0
    for i, t in enumerate(texts):
        L = len(t)
        if L > max_total_chars:
            # Shouldn't happen after splitting, but guard anyway
            if start < i:
                slices.append((start, i))
            slices.append((i, i + 1))
            start = i + 1
            cur_len = 0
        else:
            if cur_len + L > max_total_chars:
                slices.append((start, i))
                start = i
                cur_len = L
            else:
                cur_len += L
    if start < len(texts):
        slices.append((start, len(texts)))
    return slices

def read_pool(input_file: str, max_hnegs: int) -> List[Dict[str, Any]]:
    """
    Read the gz JSONL file; return full pool of usable examples with capped hard negatives.
    Each pool item has: query_en, positive_en, hard_negs_en (list, len<=max_hnegs)
    """
    pool = []
    with gzip.open(input_file, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "query" not in item or "positive_passages" not in item:
                continue
            if not item["positive_passages"]:
                continue

            pos = item["positive_passages"][0]
            pos_text = pos["text"] if isinstance(pos, dict) and "text" in pos else (pos if isinstance(pos, str) else None)
            if not pos_text:
                continue

            hnegs_raw = item.get("negative_passages", []) or item.get("hard_negative_passages", []) or []
            hard_negs = []
            for hn in hnegs_raw:
                t = hn["text"] if isinstance(hn, dict) and "text" in hn else (hn if isinstance(hn, str) else None)
                if t:
                    hard_negs.append(t)
                if len(hard_negs) >= max_hnegs:
                    break

            pool.append(
                {
                    "query_en": item["query"],
                    "positive_en": pos_text,
                    "hard_negs_en": hard_negs,  # may be empty; in-batch negatives will still help
                }
            )
    return pool


def load_or_create_sample(
    input_file: str,
    sample_file: str,
    n_samples: int,
    max_hnegs: int,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:

    if os.path.exists(sample_file):
        print(f"Loading existing sample from {sample_file}")
        with open(sample_file, "r", encoding="utf-8") as f:
            sampled = [json.loads(line) for line in f]

        if sampled and "query_en" in sampled[0]:
            return sampled
        else:
            raise RuntimeError(
                f"{sample_file} exists but has old schema. Delete it and rerun to regenerate."
            )

    print("Reading pool and sampling...")
    pool = read_pool(input_file, max_hnegs)
    print(f"Total usable pool: {len(pool)}")
    if len(pool) < n_samples:
        print(f"⚠️ Requested {n_samples} but only {len(pool)} available. Using all.")
        n_samples = len(pool)

    random.seed(seed)
    sampled = random.sample(pool, n_samples)

    with open(sample_file, "w", encoding="utf-8") as f:
        for ex in sampled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved sample to {sample_file}")
    return sampled


def save_checkpoint(rows: List[Dict[str, Any]], out_parquet: str, out_jsonl: str):
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def maybe_resume(out_parquet: str, out_jsonl: str) -> List[Dict[str, Any]]:

    if os.path.exists(out_parquet):
        print(f"Resuming from {out_parquet}")
        df = pd.read_parquet(out_parquet)
        return df.to_dict("records")
    if os.path.exists(out_jsonl):
        print(f"Resuming from {out_jsonl}")
        with open(out_jsonl, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return []



# Batch builder
def build_translation_batches(sampled: List[Dict[str, Any]], start_idx: int, batch_examples: int) -> List[List[int]]:

    batches = []
    n = len(sampled)
    i = start_idx
    while i < n:
        batch = list(range(i, min(i + batch_examples, n)))
        batches.append(batch)
        i += batch_examples
    return batches

def main():
    parser = argparse.ArgumentParser(description="Build RU version of tevatron-msmarco-aug with capped hard negatives.")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Path to msmarco-passage-aug/train.jsonl.gz")
    parser.add_argument("--sample_file", default=DEFAULT_SAMPLE_FILE, help="Where to store sampled EN subset (JSONL)")
    parser.add_argument("--out_parquet", default=DEFAULT_OUT_PARQUET)
    parser.add_argument("--out_jsonl", default=DEFAULT_OUT_JSONL)
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--max_hnegs", type=int, default=DEFAULT_MAX_HNEGS)
    parser.add_argument("--batch_examples", type=int, default=DEFAULT_BATCH_EXAMPLES)
    parser.add_argument("--checkpoint_interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--yandex_api_key", required=True)
    parser.add_argument("--yandex_folder_id", required=True)
    parser.add_argument("--cooldown_seconds", type=int, default=DEFAULT_COOLDOWN_SECONDS)
    args = parser.parse_args()

    sampled = load_or_create_sample(
        input_file=args.input,
        sample_file=args.sample_file,
        n_samples=args.n_samples,
        max_hnegs=args.max_hnegs,
        seed=args.seed,
    )

    out_rows = maybe_resume(args.out_parquet, args.out_jsonl)
    start_idx = len(out_rows)
    print(f"➡️  Already translated {start_idx}/{len(sampled)} examples")

    if start_idx >= len(sampled):
        print("✅ Nothing to do.")
        sys.exit(0)

    batches = build_translation_batches(sampled, start_idx, args.batch_examples)

    pbar = tqdm(total=len(sampled) - start_idx, desc="Translating examples")
    for batch_indices in batches:
        flat_texts: List[str] = []
        segment_map: List[Tuple[int, str, int | None]] = []  # (example_index, segment_type, pos_in_hneg)

        for idx in batch_indices:
            ex = sampled[idx]
            # query
            flat_texts.append(ex["query_en"])
            segment_map.append((idx, "query", None))
            # positive
            flat_texts.append(ex["positive_en"])
            segment_map.append((idx, "positive", None))
            # hard negatives
            for k, hn in enumerate(ex.get("hard_negs_en", [])):
                flat_texts.append(hn)
                segment_map.append((idx, "hard_neg", k))

        exploded_texts, rebuild_map = _explode_texts(flat_texts)
        slices = _pack_by_total_chars(exploded_texts, MAX_TOTAL_CHARS_PER_CALL)

        exploded_translations: List[str] = []
        for (a, b) in slices:
            sub = exploded_texts[a:b]
            sub_out = translate_batch_yandex(
                sub,
                api_key=args.yandex_api_key,
                folder_id=args.yandex_folder_id,
                cooldown_seconds=args.cooldown_seconds,
            )
            exploded_translations.extend(sub_out)

        translated: List[str] = []
        cursor = 0
        for (_, n_parts) in rebuild_map:
            parts = exploded_translations[cursor:cursor + n_parts]
            translated.append(" ".join(parts))
            cursor += n_parts

        assert len(translated) == len(flat_texts), (len(translated), len(flat_texts))

        partial_ru: Dict[int, Dict[str, Any]] = {}
        for (ex_idx, seg_type, pos), ru_text in zip(segment_map, translated):
            if ex_idx not in partial_ru:
                partial_ru[ex_idx] = {"query_ru": None, "positive_ru": None, "hard_negs_ru": {}}
            if seg_type == "query":
                partial_ru[ex_idx]["query_ru"] = ru_text
            elif seg_type == "positive":
                partial_ru[ex_idx]["positive_ru"] = ru_text
            elif seg_type == "hard_neg":
                partial_ru[ex_idx]["hard_negs_ru"][pos] = ru_text

        for ex_idx in sorted(partial_ru.keys()):
            ex = sampled[ex_idx]
            ru_pack = partial_ru[ex_idx]
            hard_negs_ru = [ru_pack["hard_negs_ru"][i] for i in range(len(ex.get("hard_negs_en", [])))]

            row = {
                "query_en": ex["query_en"],
                "positive_en": ex["positive_en"],
                "hard_negs_en": ex.get("hard_negs_en", []),
                "query_ru": ru_pack["query_ru"],
                "positive_ru": ru_pack["positive_ru"],
                "hard_negs_ru": hard_negs_ru,
            }
            out_rows.append(row)
            pbar.update(1)

            if len(out_rows) % args.checkpoint_interval == 0:
                save_checkpoint(out_rows, args.out_parquet, args.out_jsonl)
                print(f"💾 Checkpoint saved at {len(out_rows)} examples")


    pbar.close()
    save_checkpoint(out_rows, args.out_parquet, args.out_jsonl)
    print(f"✅ Done. Wrote {len(out_rows)} rows to:\n  - {args.out_parquet}\n  - {args.out_jsonl}")
    print("Schema per row:")
    print("  query_en:str, positive_en:str, hard_negs_en:list[str], query_ru:str, positive_ru:str, hard_negs_ru:list[str]")


if __name__ == "__main__":
    main()
