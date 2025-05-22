# evaluate_bertscore.py

import os
from bert_score import score
import pandas as pd

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

#chat helped with this one
def main():
    results = []
    for model_name, sys_dir in SYSTEM_DIRS.items():
        for style in STYLES:
            sys_texts = []
            ref_texts = []
            for talk in TALKS:
                sys_path = os.path.join(sys_dir, f"{talk}_{style}.txt")
                ref_path = os.path.join(REFERENCE_DIR, f"ref_{talk}.txt")
                if not os.path.exists(sys_path) or not os.path.exists(ref_path):
                    print(f"Skipping missing file(s): {sys_path}, {ref_path}")
                    continue
                sys_texts.append(load_text(sys_path))
                ref_texts.append(load_text(ref_path))

            if not sys_texts:
                continue

            # compute BERTScore over the list
            P, R, F1 = score(
                sys_texts, 
                ref_texts, 
                lang="en", 
                rescale_with_baseline=True
            )
            results.append({
                "Model": model_name,
                "Style": style,
                "BERTScore P": float(P.mean()),
                "BERTScore R": float(R.mean()),
                "BERTScore F1": float(F1.mean())
            })

    #build a dataframe
    df = pd.DataFrame(results).round(3)
    df = df.set_index(["Model","Style"])
    print("\n=== Average BERTScore ===\n")
    print(df)
    #save csv
    df.to_csv("bertscore_results.csv")
    print("\nResults saved to bertscore_results.csv")

if __name__ == "__main__":
    main()
