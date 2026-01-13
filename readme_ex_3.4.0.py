import csv
import os
from pyannote.audio import Pipeline
import torch

# Load pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ.get("HF_TOKEN"))

# Send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# Apply pretrained pipeline
audio_path = "/home/brendanoconnor/gs_imports/roformer_voice_separated_upto_step5/audio/id00570/edd38ac554a9899aca38c43ed81107dc/00001.wav"
diarization = pipeline(audio_path)

# Write results to CSV
output_csv = "diarization_results.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["speaker_class", "start_time", "stop_time"])
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        writer.writerow([speaker, turn.start, turn.end])
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

print(f"\nResults saved to {output_csv}")


