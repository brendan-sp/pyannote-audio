import torch
import os
import sys
import pdb
import argparse
import pandas as pd
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from custom_speaker_embedding import MyCustomSpeakerEmbedding
# Add both paths so voice_bio_service AND voice_bio are importable



def load_model(custom_model_path):
    # Community-1 open-source speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=os.environ.get("HF_TOKEN"))

    # send pipeline to GPU (when available)
    pipeline.to(torch.device("cuda"))

    # Replace the embedding model with yours
    pipeline._embedding = MyCustomSpeakerEmbedding(
        model_path=custom_model_path,
        device=pipeline.device
    )

    pipeline.clustering = AgglomerativeClustering(metric="cosine")

    # Set the clustering hyperparameters
    pipeline.clustering.threshold = 1.0  # tune this (0.0-2.0), originally 0.7
    pipeline.clustering.method = "centroid"  # or "average", "complete", etc.
    pipeline.clustering.min_cluster_size = 10

    # Also update the audio resampler to match your sample rate
    from pyannote.audio.core.io import Audio
    pipeline._audio = Audio(
        sample_rate=pipeline._embedding.sample_rate, 
        mono="downmix"
    )

    return pipeline

def get_df_diarization(pipeline, audio_path):
    

    # apply pretrained pipeline (with optional progress hook)
    with ProgressHook() as hook:
        output = pipeline(
            audio_path,
            hook=hook,
            min_speakers=1, # 0 singers will be handled separately
            max_speakers=4,
        )  # runs locally

    # Build DataFrame from diarization results
    rows = [
        {"speaker_class": speaker, "start_time": turn.start, "stop_time": turn.end}
        for turn, speaker in output.speaker_diarization
    ]
    df = pd.DataFrame(rows)
    
    # Write results to CSV

    return df


# Step 2: Find all overlapping (multiclass) segments >= 0.5s
def find_overlaps(df):
    """Find all overlapping time segments between different speakers."""
    overlaps = []
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i >= j:
                continue
            # Check if segments overlap
            overlap_start = max(row_i['start_time'], row_j['start_time'])
            overlap_end = min(row_i['stop_time'], row_j['stop_time'])
            if overlap_start < overlap_end:
                # Collect all speakers involved
                speakers = sorted(set([row_i['speaker_class'], row_j['speaker_class']]))
                overlaps.append({
                    'start_time': overlap_start,
                    'stop_time': overlap_end,
                    'duration': overlap_end - overlap_start,
                    'speakers': speakers
                })
    return pd.DataFrame(overlaps)


# Step 3: Split segments around multiclass gaps to create singleclass_segments
def merge_consecutive_segments(df):
    """
    Merge segments of the same speaker that occur consecutively
    (i.e., no other speaker appears between them), regardless of gap duration.
    """
    if len(df) == 0:
        return df.copy()
    
    # Sort by start time to get chronological order
    df = df.sort_values('start_time').reset_index(drop=True)
    
    merged_rows = []
    current_row = None
    
    for idx, row in df.iterrows():
        if current_row is None:
            current_row = row.to_dict()
            continue
        
        # Check if same speaker (consecutive in time, no other speaker in between)
        if row['speaker_class'] == current_row['speaker_class']:
            # Merge: extend current segment to include this one
            current_row['stop_time'] = row['stop_time']
        else:
            # Different speaker, save current and start new
            merged_rows.append(current_row)
            current_row = row.to_dict()
    
    # Don't forget the last segment
    if current_row is not None:
        merged_rows.append(current_row)
    
    return pd.DataFrame(merged_rows)


def create_singleclass_segments(df, multiclass_df, min_duration=0.5):
    """
    Split original segments around multiclass (overlap) regions.
    Merge same-speaker segments that occur consecutively (no other speaker in between).
    
    Returns DataFrame with only single-speaker segments, gaps where multiclass occurred.
    """
    if len(multiclass_df) == 0:
        # No multiclass segments, just merge consecutive same-speaker segments
        result_df = merge_consecutive_segments(df)
        result_df = result_df.sort_values('start_time').reset_index(drop=True)
        return result_df
    
    # Collect all multiclass regions to exclude
    multiclass_regions = list(zip(multiclass_df['start_time'], multiclass_df['stop_time']))
    
    # Merge overlapping/adjacent multiclass regions
    multiclass_regions.sort(key=lambda x: x[0])
    merged_regions = []
    for m_start, m_end in multiclass_regions:
        if merged_regions and m_start <= merged_regions[-1][1]:
            merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], m_end))
        else:
            merged_regions.append((m_start, m_end))
    
    # Process each segment and split around multiclass regions
    new_rows = []
    for idx, row in df.iterrows():
        start, end = row['start_time'], row['stop_time']
        speaker = row['speaker_class']
        
        # Find all multiclass regions that intersect this segment
        intersecting = []
        for m_start, m_end in merged_regions:
            if m_start < end and m_end > start:
                intersecting.append((max(m_start, start), min(m_end, end)))
        
        if not intersecting:
            # No multiclass regions affect this segment, keep as is
            new_rows.append({
                'speaker_class': speaker,
                'start_time': start,
                'stop_time': end
            })
            continue
        
        # Sort intersecting regions by start time
        intersecting.sort(key=lambda x: x[0])
        
        # Create new segments by cutting out the multiclass regions
        current_start = start
        for m_start, m_end in intersecting:
            if current_start < m_start:
                # Add segment before multiclass region
                new_rows.append({
                    'speaker_class': speaker,
                    'start_time': current_start,
                    'stop_time': m_start
                })
            current_start = m_end
        
        # Add final segment after last multiclass region
        if current_start < end:
            new_rows.append({
                'speaker_class': speaker,
                'start_time': current_start,
                'stop_time': end
            })
    
    result_df = pd.DataFrame(new_rows)
    
    # Merge same-speaker segments that occur consecutively
    result_df = merge_consecutive_segments(result_df)
    
    # Remove any segments that are too short
    result_df['duration'] = result_df['stop_time'] - result_df['start_time']
    result_df = result_df[result_df['duration'] >= min_duration]
    result_df = result_df.drop(columns=['duration'])
    
    # Sort by start time
    result_df = result_df.sort_values('start_time').reset_index(drop=True)
    
    return result_df


def filter_main_singer(
    df,
    min_seg_dur=0.5,
    verbose=False,
    ):

    df['duration'] = df['stop_time'] - df['start_time']
    df_filtered = df[df['duration'] >= min_seg_dur].copy()
    df_filtered = df_filtered.drop(columns=['duration']).reset_index(drop=True)
    overlaps_df = find_overlaps(df_filtered)

    # Filter to only >= 0.5s overlaps
    if len(overlaps_df) > 0:
        multiclass_segments = overlaps_df[overlaps_df['duration'] >= min_seg_dur].copy()
        multiclass_segments = multiclass_segments.reset_index(drop=True)
    else:
        multiclass_segments = pd.DataFrame(columns=['start_time', 'stop_time', 'duration', 'speakers'])


    singleclass_segments = create_singleclass_segments(
        df_filtered, multiclass_segments, 
        min_duration=min_seg_dur
    )
    if verbose:
        print(f"Original segments: {len(df)}")
        print(f"After removing < {min_seg_dur}s: {len(df_filtered)}")
        print(f"Removed: {len(df) - len(df_filtered)} short segments")
        print(f"Initially found {len(overlaps_df)} overlapping segments")
        print(f"Multiclass segments >= {min_seg_dur}s: {len(multiclass_segments)}")
        print(f"Singleclass segments: {len(singleclass_segments)}")

    return singleclass_segments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--min_seg_dur", type=float, default=0.5)
    parser.add_argument("--custom_model_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    ds_path = args.ds_path
    pipeline = load_model(custom_model_path=args.custom_model_path)
    if os.path.isdir(ds_path):
        file_list = [os.path.join(ds_path, f) for f in os.listdir(ds_path) if f.endswith(".wav") and not f.startswith(".") and not os.path.isdir(os.path.join(ds_path, f))]
    elif os.path.isfile(ds_path):
        file_list = [ds_path]
        ds_path = os.path.dirname(ds_path)
    else:
        raise ValueError(f"Invalid dataset path: {ds_path}")    

    diar_csvs_dir = os.path.join(ds_path, "single_singer_diarizations")

    for audio_path in file_list:
        track_name = os.path.splitext(os.path.basename(audio_path))[0]
        dst_csv_path = os.path.join(diar_csvs_dir, f"{track_name}.csv")
        df = get_df_diarization(pipeline, audio_path)
        singleclass_segments = filter_main_singer(df, min_seg_dur=args.min_seg_dur, verbose=True)
        singleclass_segments.to_csv(dst_csv_path, index=False)
        print(f"Saved singleclass segments to {dst_csv_path}")

if __name__ == "__main__":

    main()