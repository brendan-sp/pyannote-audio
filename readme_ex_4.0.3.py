import torch
import csv
import pdb
import os
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


# Community-1 open-source speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ.get("HF_TOKEN"))

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline (with optional progress hook)
with ProgressHook() as hook:
    output = pipeline(
        "/home/brendanoconnor/gs_imports/roformer_voice_separated_upto_step5/audio/id00570/edd38ac554a9899aca38c43ed81107dc/00001.wav",
        hook=hook
    )  # runs locally

# Write results to CSV
output_csv = "diarization_results_4.0.3.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["speaker_class", "start_time", "stop_time"])
    for turn, speaker in output.speaker_diarization:
        writer.writerow([speaker, turn.start, turn.end])
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

print(f"\nResults saved to {output_csv}")