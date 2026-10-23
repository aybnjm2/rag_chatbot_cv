import csv
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import answer_cv_only, build_store_from_pdf  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CV_DIR = EVAL_DIR / "cvs"
QUESTIONS_FILE = EVAL_DIR / "questions.csv"
OUT_CSV = EVAL_DIR / "comparison.csv"
OUT_MD = EVAL_DIR / "comparison.md"

PAUSE_SECONDS = 1.0

def is_refusal(answer):
    """Detects a refusal, even if the LLM rephrases slightly."""
    return answer.strip().lower().startswith("je ne peux pas répondre")


def parse_keywords(raw):
    return [k.strip().lower() for k in (raw or "").split("|") if k.strip()]


def contains_all(text, keywords):
    text = text.lower()
    return all(k in text for k in keywords)

# params
CONFIGS = [
    {"name": "chunk 600 / k=4",  "chunk_size": 600,  "overlap": 90,  "k": 4, "min_score": 0.35},
    {"name": "chunk 1000 / k=4", "chunk_size": 1000, "overlap": 150, "k": 4, "min_score": 0.35},
    {"name": "chunk 1000 / k=6", "chunk_size": 1000, "overlap": 150, "k": 6, "min_score": 0.35},
    # Threshold sweep (cheap: reuses the same stores)
    {"name": "chunk 1000 / k=4 / min 0.25", "chunk_size": 1000, "overlap": 150, "k": 4, "min_score": 0.25},
    {"name": "chunk 1000 / k=4 / min 0.45", "chunk_size": 1000, "overlap": 150, "k": 4, "min_score": 0.45},
]


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


_store_cache = {}


def get_store(cv_file, chunk_size, overlap):
    """Index each (cv, chunk_size, overlap) only once, across configs."""
    key = (cv_file, chunk_size, overlap)
    if key not in _store_cache:
        stem = re.sub(r"[^a-zA-Z0-9]", "", Path(cv_file).stem)[:20] or "cv"
        name = f"cmp_{stem}_{chunk_size}_{overlap}"
        print(f"  Indexing {cv_file} (chunk={chunk_size}, overlap={overlap})...")
        _store_cache[key] = build_store_from_pdf(
            (CV_DIR / cv_file).read_bytes(),
            collection_name=name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    return _store_cache[key]


def fmt(n, d):
    return "n/a" if d == 0 else f"{n}/{d} ({100 * n / d:.0f}%)"


def evaluate_config(cfg, questions):
    hits = hit_total = 0          # retrieval hit (answerable only)
    kw_ok = kw_total = 0          # keywords in the answer
    not_refused = ans_total = 0   # answerable questions not refused
    refusal_ok = unans_total = 0  # unanswerable questions correctly refused
    latencies, errors = [], 0

    for i, q in enumerate(questions, start=1):
        store = get_store(q["cv_file"], cfg["chunk_size"], cfg["overlap"])
        answerable = q["answerable"].strip().lower() == "oui"
        keywords = parse_keywords(q.get("expected_keywords", ""))

        # Retrieval quality, measured on raw top-k (no threshold)
        if answerable and keywords:
            raw = store.similarity_search_with_relevance_scores(
                q["question"], k=cfg["k"]
            )
            raw_context = "\n".join(d.page_content for d, _ in raw)
            hit_total += 1
            hits += contains_all(raw_context, keywords)

        # Full pipeline, timed
        start = time.perf_counter()
        try:
            answer, _ = answer_cv_only(
                q["question"], store, k=cfg["k"], min_score=cfg["min_score"]
            )
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"    [error] {q['question'][:50]}: {e}")
            continue
        latencies.append(time.perf_counter() - start)

        refused = is_refusal(answer)
        if answerable:
            ans_total += 1
            not_refused += not refused
            if keywords:
                kw_total += 1
                kw_ok += contains_all(answer, keywords)
        else:
            unans_total += 1
            refusal_ok += refused

        print(f"    [{i}/{len(questions)}] {q['question'][:60]}")
        time.sleep(PAUSE_SECONDS)

    return {
        "config": cfg["name"],
        "retrieval_hit": fmt(hits, hit_total),
        "keywords_ok": fmt(kw_ok, kw_total),
        "answered_when_possible": fmt(not_refused, ans_total),
        "correct_refusal": fmt(refusal_ok, unans_total),
        "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else "n/a",
        "errors": errors,
    }


def to_markdown(results):
    header = ("| Config | Retrieval hit | Correct refusal | Keywords OK "
              "| Answered when possible | Avg latency (s) |")
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['retrieval_hit']} | {r['correct_refusal']} "
            f"| {r['keywords_ok']} | {r['answered_when_possible']} "
            f"| {r['avg_latency_s']} |"
        )
    return "\n".join(lines)


def cleanup():
    for store in _store_cache.values():
        try:
            store.delete_collection()
        except Exception:  # noqa: BLE001
            pass


def main():
    questions = load_questions()
    n_ans = sum(q["answerable"].strip().lower() == "oui" for q in questions)
    print(f"{len(questions)} questions ({n_ans} answerable), {len(CONFIGS)} configs\n")

    results = []
    try:
        for cfg in CONFIGS:
            print(f"== {cfg['name']} ==")
            results.append(evaluate_config(cfg, questions))
    finally:
        cleanup()

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    table = to_markdown(results)
    header = f"Test set: {len(questions)} questions ({n_ans} answerable)\n\n"
    OUT_MD.write_text(header + table + "\n", encoding="utf-8")

    print("\n" + header + table)
    print(f"\nSaved: {OUT_MD} and {OUT_CSV}")


if __name__ == "__main__":
    main()