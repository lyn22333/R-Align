import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


RESULT_DIR = Path(__file__).resolve().parent / "result_metarm"
OUT_CSV = Path(__file__).resolve().parent / "results_summary.csv"


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class FileStats:
    path: Path
    bench_name: str = ""
    model_name: str = ""
    total: int = 0
    label_correct: int = 0
    metarm_call_fail: int = 0
    metarm_parse_fail: int = 0
    metarm_correct: int = 0
    metarm_incorrect: int = 0

    def label_acc(self) -> Optional[float]:
        return (self.label_correct / self.total) if self.total else None

    def metarm_ok(self) -> int:
        return self.metarm_correct + self.metarm_incorrect

    def metarm_acc(self) -> Optional[float]:
        denom = self.metarm_ok()
        if denom == 0:
            return None
        return self.metarm_correct / denom

    def spurious_correctness(self) -> Optional[float]:
        acc = self.metarm_acc()
        if acc is None:
            return None
        return 1.0 - acc

    def fidelity_score(self) -> Optional[float]:
        denom = self.total - self.metarm_call_fail - self.metarm_parse_fail
        if denom <= 0:
            return None
        return self.metarm_correct / denom


def compute_file_stats(path: Path) -> FileStats:
    st = FileStats(path=path)
    st.bench_name, st.model_name = bench_and_model_from_filename(path)
    for item in iter_jsonl(path):
        st.total += 1
        model_pred = item.get("model_pred", None)
        gt_label = (item.get("gt_label", "") or "").strip()
        if model_pred in ("A", "B") and model_pred == gt_label:
            st.label_correct += 1

        verdict = item.get("metarm_verdict", None)
        if verdict == "CallFail":
            st.metarm_call_fail += 1
        elif verdict == "ParseFail":
            st.metarm_parse_fail += 1
        elif verdict == "Correct":
            st.metarm_correct += 1
        elif verdict == "Incorrect":
            st.metarm_incorrect += 1
    return st


def bench_and_model_from_filename(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if "__" in stem:
        bench, rest = stem.split("__", 1)
    else:
        bench, rest = stem, ""
    model = rest
    if "_metarm_" in rest:
        model = rest.split("_metarm_", 1)[0]
    return bench, model


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.2f}%"


def fmt_pct_csv(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x * 100:.4f}"


def main() -> None:
    if not RESULT_DIR.exists():
        raise FileNotFoundError(f"result dir not found: {RESULT_DIR}")

    paths = sorted(p for p in RESULT_DIR.glob("*.jsonl") if p.is_file())
    if not paths:
        print(f"[score] no files in {RESULT_DIR}")
        return

    stats = [compute_file_stats(p) for p in paths]

    benches: List[str] = sorted({st.bench_name for st in stats if st.bench_name})
    models: List[str] = sorted({st.model_name for st in stats if st.model_name})
    index: Dict[Tuple[str, str], FileStats] = {
        (st.model_name, st.bench_name): st
        for st in stats
        if st.model_name and st.bench_name
    }

    for m in models:
        print(m)
        for b in benches:
            st = index.get((m, b))
            if st is None:
                continue
            print(
                f"  {b}: label_acc={fmt_pct(st.label_acc())}   "
                f"spurious_correctness={fmt_pct(st.spurious_correctness())}   "
                f"Fidelity Score={fmt_pct(st.fidelity_score())}"
            )
        print("")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = ["model"]
    for b in benches:
        fieldnames.extend(
            [
                f"{b}__label_acc",
                f"{b}__spurious_correctness",
                f"{b}__fidelity_score",
            ]
        )
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in models:
            row: Dict[str, str] = {"model": m}
            for b in benches:
                st = index.get((m, b))
                if st is None:
                    row[f"{b}__label_acc"] = ""
                    row[f"{b}__spurious_correctness"] = ""
                    row[f"{b}__fidelity_score"] = ""
                    continue
                row[f"{b}__label_acc"] = fmt_pct_csv(st.label_acc())
                row[f"{b}__spurious_correctness"] = fmt_pct_csv(
                    st.spurious_correctness()
                )
                row[f"{b}__fidelity_score"] = fmt_pct_csv(st.fidelity_score())
            w.writerow(row)

    print(f"[score] wrote csv -> {OUT_CSV}")


if __name__ == "__main__":
    main()

