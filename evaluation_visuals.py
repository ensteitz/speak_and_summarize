# evaluate_insights_visual.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from bert_score import score

REFERENCE_DIR = "reference_summaries"
SYSTEM_DIRS = {
    "OpenAI (gpt-3.5-turbo)": "summaries_openai",
    "Local T5 (t5-small)"   : "summaries_t5_local"
}
TALKS  = ["education", "business", "psychology"]
STYLES = ["zero_shot", "role_based", "chain_of_thought"]


def load_text(path): 
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

records = []
for model_name, sys_dir in SYSTEM_DIRS.items():
    for style in STYLES:
        for talk in TALKS:
            sys_fn = f"{talk}_{style}.txt"
            sys_path = os.path.join(sys_dir, sys_fn)
            ref_path = os.path.join(REFERENCE_DIR, f"ref_{talk}.txt")
            if not (os.path.exists(sys_path) and os.path.exists(ref_path)):
                continue
            system = load_text(sys_path)
            reference = load_text(ref_path)
            # Compute ROUGE
            rouge_scores = rouge_scorer_obj.score(reference, system)
            r1 = rouge_scores["rouge1"].fmeasure
            r2 = rouge_scores["rouge2"].fmeasure
            rl = rouge_scores["rougeL"].fmeasure
            # Compute BERTScore
            P, R, F1 = score([system], [reference], lang='en', rescale_with_baseline=True)
            bert_f1 = F1[0].item()
            # Summary length
            length = len(system.split())
            records.append({
                "Model": model_name,
                "Style": style,
                "Talk": talk,
                "ROUGE-1": r1,
                "ROUGE-2": r2,
                "ROUGE-L": rl,
                "BERT-F1": bert_f1,
                "Length": length
            })

#create dataframe with pandas
df = pd.DataFrame.from_records(records)

#heatmap
corr = df[["ROUGE-1","ROUGE-2","ROUGE-L","BERT-F1","Length"]].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
plt.title("Correlation Matrix of Metrics & Summary Length")
plt.tight_layout()
plt.savefig("metric_correlation.png")
plt.show()

# bar plots
df_melt = df.melt(id_vars=["Model","Style"], value_vars=["ROUGE-1","ROUGE-2","ROUGE-L","BERT-F1"], 
                  var_name="Metric", value_name="Score")

g = sns.catplot(
    data=df_melt, x="Style", y="Score", hue="Model", col="Metric",
    kind="bar", height=4, aspect=0.8
)
g.set_xticklabels(rotation=45)
plt.tight_layout()
plt.savefig("metrics_barplot.png")
plt.show()
