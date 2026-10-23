import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import answer_cv_only, build_store_from_pdf, TOP_K  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


def main():
    with open(EVAL_DIR / "questions.csv", encoding="utf-8-sig") as f:
        questions = list(csv.DictReader(f))

    
    stores = {} # Un store par CV, construit une seule fois
    for i, cv_file in enumerate(sorted({q["cv_file"] for q in questions})):
        print(f"Indexation de {cv_file}...")
        pdf_bytes = (EVAL_DIR / "cvs" / cv_file).read_bytes()
        stores[cv_file] = build_store_from_pdf(pdf_bytes, collection_name=f"eval_{i}")

    rows = []
    for q in questions:
        store = stores[q["cv_file"]]
        answerable = q["answerable"].strip().lower() == "oui"
        keywords = [k.strip().lower() for k in q["expected_keywords"].split("|") if k.strip()]

        # meilleur score de similarite (sert a calibrer MIN_SCORE)
        results = store.similarity_search_with_relevance_scores(q["question"], k=TOP_K)
        top_score = max((score for _, score in results), default=0.0)

        # pipeline complet, chronometre
        start = time.perf_counter()
        answer, _ = answer_cv_only(q["question"], store)
        latency = time.perf_counter() - start

        refused = answer.strip().lower().startswith("je ne peux pas répondre")

        if answerable:
            ok = (not refused) and all(k in answer.lower() for k in keywords)
        else:
            ok = refused

        rows.append({
            "question": q["question"],
            "answerable": q["answerable"],
            "answer": answer.replace("\n", " "),
            "top_score": round(top_score, 3),
            "ok": ok,
            "latency_s": round(latency, 2),
            "manual_check": "",  # a remplir a la main : ok / partiel / faux
        })
        print(f"{'OK ' if ok else 'ERR'} {latency:.1f}s  {q['question'][:60]}")

    with open(EVAL_DIR / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Résumé
    ans = [r for r in rows if r["answerable"] == "oui"]
    unans = [r for r in rows if r["answerable"] != "oui"]

    print("\n===== RESUMe =====")
    print(f"Repondables bien traitees    : {sum(r['ok'] for r in ans)}/{len(ans)}")
    print(f"Non répondables bien refusées: {sum(r['ok'] for r in unans)}/{len(unans)}")
    print(f"Latence moyenne              : {sum(r['latency_s'] for r in rows) / len(rows):.2f} s")

    if ans and unans:
        print("\n===== CALIBRATION DE MIN_SCORE =====")
        print(f"Plus bas score (répondables)     : {min(r['top_score'] for r in ans)}")
        print(f"Plus haut score (non répondables): {max(r['top_score'] for r in unans)}")
        print("Si le premier est supérieur au second, choisis un seuil entre les deux.")


if __name__ == "__main__":
    main()