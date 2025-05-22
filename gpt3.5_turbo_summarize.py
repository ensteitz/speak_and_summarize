# run_openai_summaries.py

import os
import openai

# load OpenAI API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

#map each transcript to its filename (in the audio_files folder)
TRANSCRIPTS = {
    "education":  os.path.join("audio_files", "education.txt"),
    "psychology": os.path.join("audio_files", "psychology.txt"),
    "business":   os.path.join("audio_files", "business.txt")
}

# Prompt templates!
PROMPTS = {
    "zero_shot": """Please summarize the following transcript in 150 words or fewer, capturing the main points clearly and concisely:

{transcript}""",
    "role_based": """You are an expert audio summarization specialist. Review the following transcript and produce a concise summary (no more than 150 words) that highlights the key themes, speaker insights, and any action items:

{transcript}""",
    "chain_of_thought": """Let's work through this systematically:
1. Identify the primary topics covered.
2. Note any supporting details or examples for each topic.
3. Outline the main conclusions or takeaways.
4. Finally, provide a concise summary in 150 words or fewer.

Transcript:
{transcript}"""
}

#define the output directory for new OpenAI summaries
OUTPUT_DIR = "summaries_openai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def call_openai(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def main():
    for key, path in TRANSCRIPTS.items():
        if not os.path.isfile(path):
            print(f"Transcript file not found: {path}")
            continue
        transcript = open(path, "r", encoding="utf-8").read()
        
        for strat, template in PROMPTS.items():
            prompt = template.format(transcript=transcript)
            summary = call_openai(prompt)
            
            out_path = os.path.join(OUTPUT_DIR, f"{key}_{strat}.txt")
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(summary)
            print(f"Saved summary to {out_path}")

if __name__ == "__main__":
    main()