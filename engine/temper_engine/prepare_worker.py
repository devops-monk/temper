"""Dataset preparation worker: `python -m temper_engine.prepare_worker <prepare.json>`.

Maps columns to chat turns, cleans and splits the whole dataset, and writes Temper's standard JSONL
(`messages` or `text` rows) plus an `.eval.jsonl` split that training picks up by itself.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from . import formats
from .jobs import emit_from_worker as emit


def read_rows(source: str, split: str, config: str | None = None) -> list[dict]:
    path = Path(source).expanduser()
    suffix = path.suffix.lower()
    if path.is_file() and suffix in (".csv", ".json", ".jsonl"):
        with open(path, encoding="utf-8") as f:
            if suffix == ".csv":
                return list(csv.DictReader(f))
            if suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Preparing Hugging Face and Parquet datasets needs training support. Add it in Settings.") from None
    if path.is_file():
        return list(load_dataset("parquet", data_files=str(path), split="train"))
    return list(load_dataset(source, config, split=split or "train"))


def write(path: Path, convs: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(formats.record(c), ensure_ascii=False) + "\n" for c in convs)


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    try:
        emit("phase", name="Reading the dataset")
        rows = read_rows(cfg["source"], cfg.get("split", "train"), cfg.get("config"))
        emit("progress", value=0.5)
        emit("phase", name="Cleaning")
        opts = formats.CleanOptions(**cfg["options"])
        result = formats.clean(rows, cfg["format"], cfg.get("mapping"), cfg.get("system_prompt"), opts)
        if not result.train:
            raise RuntimeError(f"No rows left after cleaning: {json.dumps(result.stats)}")
        out = Path(cfg["output"])
        write(out, result.train)
        eval_path = out.with_suffix(".eval.jsonl")
        if result.eval:
            write(eval_path, result.eval)
        emit("result", output=str(out), eval_output=str(eval_path) if result.eval else None, stats=result.stats)
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
