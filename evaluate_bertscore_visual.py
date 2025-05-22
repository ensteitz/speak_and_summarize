# evaluate_bertscore_visual.py

import os
from bert_score import score
import pandas as pd
import matplotlib.pyplot as plt

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

def main():
    results = []
    #compute BERTScore for each talk/style/model, chat helped with this one
    for model_name, sys_dir in SYSTEM_DIRS.items():
        for style in STYLES:
            for talk in TALKS:
                sys_path = os.path.join(sys_dir, f"{talk}_{style}.txt")
                ref_path = os.path.join(REFERENCE_DIR, f"ref_{talk}.txt")
                if not os.path.exists(sys_path) or not os.path.exists(ref_path):
                    continue
                system  = load_text(sys_path)
                reference = load_text(ref_path)
                P, R, F1 = score([system], [reference], lang='en', rescale_with_baseline=True)
                results.append({
                    "Model": model_name,
                    "Style": style,
                    "Precision": P[0].item(),
                    "Recall": R[0].item(),
                    "F1": F1[0].item()
                })

    #create dataframe and average thescores
    df = pd.DataFrame(results)
    df_avg = df.groupby(["Model", "Style"]).mean().reset_index()

    #print table to terminal
    print("\n=== BERTScore Averages ===")
    print(df_avg.round(3))

    df_vis = df_avg.set_index(["Model", "Style"])
    df_vis = df_vis.round(3)

    #table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(
        cellText=df_vis.values,
        colLabels=df_vis.columns,
        rowLabels=[f"{m} / {s}" for m, s in df_vis.index],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig("bertscore_table.png")
    plt.show()

if __name__ == "__main__":
    main()
