import whisper
import os

# Path to folder containing my .mp3 files
audio_folder = r"C:\Users\erins\OneDrive\Desktop\Grad School Classes\715 llms seminar\Final Project\audio_files"

#load the base Whisper model
model = whisper.load_model("base")

#fpr each MP3 files, trancribe it
for file in os.listdir(audio_folder):
    if file.endswith(".mp3"):
        audio_path = os.path.join(audio_folder, file)
        print(f"Transcribing {file}...")

        #run the transcription
        result = model.transcribe(audio_path)

        # Save transcript
        txt_filename = file.replace(".mp3", ".txt")
        output_path = os.path.join(audio_folder, txt_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Saved transcript to {txt_filename}")
