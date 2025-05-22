# run_t5_local_summaries.py

import os
from transformers import pipeline

# Map each transcript to its filename
TRANSCRIPTS = {
    "education":   os.path.join("audio_files", "education.txt"),
    "psychology":  os.path.join("audio_files", "psychology.txt"),
    "business":    os.path.join("audio_files", "business.txt")
}

# Prompt templates (we prepend a role or instruction but pipeline ignores it; pipeline uses model defaults)
PROMPTS = {
    "zero_shot":      "summarize: {transcript}",
    "role_based":     "summarize as an audio expert: {transcript}",
    "chain_of_thought": (
        "Let's break this down:\n"
        "1. List the main topics.\n"
        "2. Note key details/examples.\n"
        "3. Summarize in up to 150 words.\n\n"
        "{transcript}"
    )
}

OUTPUT_DIR = "summaries_t5_local"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load local summarizer pipeline, chat helped with this one
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt")

    for key, path in TRANSCRIPTS.items():
        if not os.path.isfile(path):
            print(f"Transcript file not found: {path}")
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Hugely long inputs need chunking
        # For simplicity, summarizer can handle ~512 tokens; we'll let it truncate if too long
        for strat, template in PROMPTS.items():
            input_text = template.format(transcript=text)
            summary_outputs = summarizer(
                input_text,
                max_length=150,
                min_length=50,
                do_sample=False
            )
            summary = summary_outputs[0]["summary_text"].strip()
            
            out_path = os.path.join(OUTPUT_DIR, f"{key}_{strat}.txt")
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(summary)
            print(f"Saved summary to {out_path}")

if __name__ == "__main__":
    main()