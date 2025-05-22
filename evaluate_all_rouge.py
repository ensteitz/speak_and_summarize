import os
import csv
from rouge_score import rouge_scorer

# --- CONFIGURATION ---
REFERENCE_DIR = "reference_summaries"
SYSTEM_DIRS = {
    "OpenAI (gpt-3.5-turbo)": "summaries_openai",
    "Local T5 (t5-small)":   "summaries_t5_local"
}
TALKS  = ["education", "business", "psychology"]
STYLES = ["zero_shot", "role_based", "chain_of_thought"]

# Initialize scorer
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], 
    use_stemmer=True
)

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    #accumulate scores
    scores = {model: {style: [] for style in STYLES} for model in SYSTEM_DIRS}

    # loop through each system's directory
    for model_name, sys_dir in SYSTEM_DIRS.items():
        for style in STYLES:
            for talk in TALKS:
                sys_path = os.path.join(sys_dir, f"{talk}_{style}.txt")
                ref_path = os.path.join(REFERENCE_DIR, f"ref_{talk}.txt")

                if not os.path.exists(sys_path) or not os.path.exists(ref_path):
                    continue

                system  = load_text(sys_path)
                reference = load_text(ref_path)
                score = scorer.score(reference, system)
                scores[model_name][style].append({
                    "rouge1": score["rouge1"].fmeasure,
                    "rouge2": score["rouge2"].fmeasure,
                    "rougeL": score["rougeL"].fmeasure
                })

    #write results to csv format
    csv_path = "rouge_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model", "Style", "ROUGE-1", "ROUGE-2", "ROUGE-L"])
        for model_name in SYSTEM_DIRS:
            for style in STYLES:
                metrics = scores[model_name][style]
                if not metrics:
                    continue
                avg1 = sum(m["rouge1"] for m in metrics) / len(metrics)
                avg2 = sum(m["rouge2"] for m in metrics) / len(metrics)
                avgl = sum(m["rougeL"] for m in metrics) / len(metrics)
                writer.writerow([model_name, style, f"{avg1:.3f}", f"{avg2:.3f}", f"{avgl:.3f}"])

    print(f"Results written to {csv_path}")

if __name__ == "__main__":
    main()
