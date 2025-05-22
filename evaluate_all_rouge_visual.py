# evaluate_all_rouge_visual.py

import os
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt

REFERENCE_DIR = "reference_summaries"
SYSTEM_DIRS = {
    "OpenAI (gpt-3.5-turbo)": "summaries_openai",
    "Local T5 (t5-small)":   "summaries_t5_local"
}
TALKS  = ["education", "business", "psychology"]
STYLES = ["zero_shot", "role_based", "chain_of_thought"]

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)

#getaverage scores
results = []
for model_name, sys_dir in SYSTEM_DIRS.items():
    for style in STYLES:
        scores = []
        for talk in TALKS:
            sys_path = os.path.join(sys_dir, f"{talk}_{style}.txt")
            ref_path = os.path.join(REFERENCE_DIR, f"ref_{talk}.txt")
            if os.path.exists(sys_path) and os.path.exists(ref_path):
                system = open(sys_path, "r", encoding="utf-8").read().strip()
                reference = open(ref_path, "r", encoding="utf-8").read().strip()
                score = scorer.score(reference, system)
                scores.append([
                    score["rouge1"].fmeasure,
                    score["rouge2"].fmeasure,
                    score["rougeL"].fmeasure
                ])
        if scores:
            avg1 = sum(s[0] for s in scores) / len(scores)
            avg2 = sum(s[1] for s in scores) / len(scores)
            avgl = sum(s[2] for s in scores) / len(scores)
            results.append({
                "Model": model_name,
                "Style": style,
                "ROUGE-1": avg1,
                "ROUGE-2": avg2,
                "ROUGE-L": avgl
            })

#pandas dataframe
df = pd.DataFrame(results)
df.set_index(["Model", "Style"], inplace=True)
df = df.round(3)

#table plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")
tbl = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    rowLabels=[f"{m} / {s}" for m, s in df.index],
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
plt.tight_layout()
plt.savefig("rouge_scores_table.png")
plt.show()
