"""Speaker diarization module.

Provides speaker diarization functionality using pyannote.audio with custom
RawNet3 speaker embeddings. Identifies and segments audio by speaker.
"""

import argparse
import logging
import os
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from pyannote.audio.pipelines.utils.hook import ProgressHook

from custom_speaker_embedding import MyCustomSpeakerEmbedding

# Module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clustering hyperparameters
CLUSTERING_THRESHOLD = 1.0  # Tune this (0.0-2.0), originally 0.7
CLUSTERING_METHOD = "centroid"  # Options: "average", "complete", etc.
MIN_CLUSTER_SIZE = 10

# Diarization constraints
MIN_SPEAKERS = 1
MAX_SPEAKERS = 4


class Diarizer:
    """Speaker diarization using pyannote with custom embeddings.

    Performs speaker diarization on audio files, identifying distinct speakers
    and their corresponding time segments.
    """

    def __init__(self, custom_model_path: str):
        """Initialize the diarizer with a custom embedding model.

        Args:
            custom_model_path: Path to the custom RawNet3 model checkpoint.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing Diarizer on device: %s", self.device)

        self.pipeline = self._load_pipeline(custom_model_path)
        logger.info("Diarizer initialized successfully")

    def _load_pipeline(self, custom_model_path: str) -> Pipeline:
        """Load and configure the pyannote diarization pipeline.

        Args:
            custom_model_path: Path to the custom embedding model.

        Returns:
            Configured pyannote Pipeline.
        """
        # Load community speaker diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=os.environ.get("HF_TOKEN"),
        )
        pipeline.to(self.device)

        # Replace embedding model with custom RawNet3
        pipeline._embedding = MyCustomSpeakerEmbedding(
            model_path=custom_model_path,
            device=pipeline.device,
        )

        # Configure clustering
        pipeline.clustering = AgglomerativeClustering(metric="cosine")
        pipeline.clustering.threshold = CLUSTERING_THRESHOLD
        pipeline.clustering.method = CLUSTERING_METHOD
        pipeline.clustering.min_cluster_size = MIN_CLUSTER_SIZE

        # Update audio resampler to match embedding model sample rate
        pipeline._audio = Audio(
            sample_rate=pipeline._embedding.sample_rate,
            mono="downmix",
        )

        return pipeline

    def diarize(
        self,
        audio_path: str,
        min_seg_dur: float = 0.5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file.
            min_seg_dur: Minimum segment duration in seconds.
            verbose: Whether to log detailed statistics.

        Returns:
            DataFrame with columns: speaker_class, start_time, stop_time
        """
        logger.info("Starting diarization for: %s", audio_path)

        df = self._get_raw_diarization(audio_path)
        singleclass_segments = self._filter_segments(
            df,
            min_seg_dur=min_seg_dur,
            verbose=verbose,
        )

        logger.info(
            "Diarization complete: %d segments found",
            len(singleclass_segments),
        )
        return singleclass_segments

    def _get_raw_diarization(self, audio_path: str) -> pd.DataFrame:
        """Run the diarization pipeline on an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            DataFrame with raw diarization results.
        """
        with ProgressHook() as hook:
            output = self.pipeline(
                audio_path,
                hook=hook,
                min_speakers=MIN_SPEAKERS,
                max_speakers=MAX_SPEAKERS,
            )

        rows = [
            {
                "speaker_class": speaker,
                "start_time": turn.start,
                "stop_time": turn.end,
            }
            for turn, speaker in output.speaker_diarization
        ]

        return pd.DataFrame(rows)

    def _filter_segments(
        self,
        df: pd.DataFrame,
        min_seg_dur: float = 0.5,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Filter and process diarization segments.

        Removes short segments, identifies overlaps, and creates single-speaker
        segments by excluding multi-speaker regions.

        Args:
            df: Raw diarization DataFrame.
            min_seg_dur: Minimum segment duration in seconds.
            verbose: Whether to log detailed statistics.

        Returns:
            Filtered DataFrame with single-speaker segments.
        """
        # Filter by minimum duration
        df["duration"] = df["stop_time"] - df["start_time"]
        df_filtered = df[df["duration"] >= min_seg_dur].copy()
        df_filtered = df_filtered.drop(columns=["duration"]).reset_index(drop=True)

        # Find overlapping segments
        overlaps_df = self._find_overlaps(df_filtered)

        # Filter overlaps by duration
        if len(overlaps_df) > 0:
            multiclass_segments = overlaps_df[
                overlaps_df["duration"] >= min_seg_dur
            ].copy()
            multiclass_segments = multiclass_segments.reset_index(drop=True)
        else:
            multiclass_segments = pd.DataFrame(
                columns=["start_time", "stop_time", "duration", "speakers"]
            )

        # Create single-class segments
        singleclass_segments = self._create_singleclass_segments(
            df_filtered,
            multiclass_segments,
            min_duration=min_seg_dur,
        )

        if verbose:
            logger.info("Original segments: %d", len(df))
            logger.info("After removing < %ss: %d", min_seg_dur, len(df_filtered))
            logger.info("Removed: %d short segments", len(df) - len(df_filtered))
            logger.info("Initially found %d overlapping segments", len(overlaps_df))
            logger.info(
                "Multiclass segments >= %ss: %d",
                min_seg_dur,
                len(multiclass_segments),
            )
            logger.info("Singleclass segments: %d", len(singleclass_segments))

        return singleclass_segments

    def _find_overlaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find all overlapping time segments between different speakers.

        Args:
            df: DataFrame with speaker segments.

        Returns:
            DataFrame with overlap information.
        """
        overlaps = []
        for i, row_i in df.iterrows():
            for j, row_j in df.iterrows():
                if i >= j:
                    continue

                overlap_start = max(row_i["start_time"], row_j["start_time"])
                overlap_end = min(row_i["stop_time"], row_j["stop_time"])

                if overlap_start < overlap_end:
                    speakers = sorted(
                        set([row_i["speaker_class"], row_j["speaker_class"]])
                    )
                    overlaps.append({
                        "start_time": overlap_start,
                        "stop_time": overlap_end,
                        "duration": overlap_end - overlap_start,
                        "speakers": speakers,
                    })

        return pd.DataFrame(overlaps)

    def _create_singleclass_segments(
        self,
        df: pd.DataFrame,
        multiclass_df: pd.DataFrame,
        min_duration: float = 0.5,
    ) -> pd.DataFrame:
        """Split segments around multiclass (overlap) regions.

        Args:
            df: DataFrame with speaker segments.
            multiclass_df: DataFrame with overlap regions.
            min_duration: Minimum segment duration to keep.

        Returns:
            DataFrame with single-speaker segments only.
        """
        if len(multiclass_df) == 0:
            result_df = df.copy()
            result_df = result_df.sort_values("start_time").reset_index(drop=True)
            return result_df

        # Collect and merge multiclass regions
        multiclass_regions = list(
            zip(multiclass_df["start_time"], multiclass_df["stop_time"])
        )
        merged_regions = self._merge_overlapping_regions(multiclass_regions)

        # Process each segment
        new_rows = []
        for _, row in df.iterrows():
            start, end = row["start_time"], row["stop_time"]
            speaker = row["speaker_class"]

            intersecting = self._find_intersecting_regions(
                start, end, merged_regions
            )

            if not intersecting:
                new_rows.append({
                    "speaker_class": speaker,
                    "start_time": start,
                    "stop_time": end,
                })
                continue

            # Split segment around multiclass regions
            new_rows.extend(
                self._split_segment_around_regions(speaker, start, end, intersecting)
            )

        result_df = pd.DataFrame(new_rows)

        # Filter by minimum duration
        if len(result_df) > 0:
            result_df["duration"] = result_df["stop_time"] - result_df["start_time"]
            result_df = result_df[result_df["duration"] >= min_duration]
            result_df = result_df.drop(columns=["duration"])
            result_df = result_df.sort_values("start_time").reset_index(drop=True)

        return result_df

    def _merge_overlapping_regions(
        self,
        regions: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Merge overlapping or adjacent time regions.

        Args:
            regions: List of (start, end) tuples.

        Returns:
            List of merged (start, end) tuples.
        """
        if not regions:
            return []

        sorted_regions = sorted(regions, key=lambda x: x[0])
        merged = []

        for m_start, m_end in sorted_regions:
            if merged and m_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], m_end))
            else:
                merged.append((m_start, m_end))

        return merged

    def _find_intersecting_regions(
        self,
        start: float,
        end: float,
        regions: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Find regions that intersect with a given time span.

        Args:
            start: Start time.
            end: End time.
            regions: List of (start, end) tuples to check.

        Returns:
            List of intersecting regions, clipped to the input span.
        """
        intersecting = []
        for m_start, m_end in regions:
            if m_start < end and m_end > start:
                intersecting.append((max(m_start, start), min(m_end, end)))

        return sorted(intersecting, key=lambda x: x[0])

    def _split_segment_around_regions(
        self,
        speaker: str,
        start: float,
        end: float,
        regions: List[Tuple[float, float]],
    ) -> List[dict]:
        """Split a segment by cutting out specified regions.

        Args:
            speaker: Speaker identifier.
            start: Segment start time.
            end: Segment end time.
            regions: Regions to cut out.

        Returns:
            List of segment dictionaries.
        """
        new_segments = []
        current_start = start

        for m_start, m_end in regions:
            if current_start < m_start:
                new_segments.append({
                    "speaker_class": speaker,
                    "start_time": current_start,
                    "stop_time": m_start,
                })
            current_start = m_end

        if current_start < end:
            new_segments.append({
                "speaker_class": speaker,
                "start_time": current_start,
                "stop_time": end,
            })

        return new_segments

    def extract_speaker_audio(
        self,
        audio_path: str,
        segments_df: pd.DataFrame,
        output_dir: str,
    ) -> List[str]:
        """Extract and save audio for each speaker.

        Args:
            audio_path: Path to source audio file.
            segments_df: DataFrame with speaker segments.
            output_dir: Directory to save extracted audio.

        Returns:
            List of output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        y, sr = librosa.load(audio_path, sr=None)
        output_paths = []

        singer_ids = segments_df["speaker_class"].unique()

        for singer_id in singer_ids:
            singer_segments = segments_df[
                segments_df["speaker_class"] == singer_id
            ]

            audio_chunks = []
            for _, row in singer_segments.iterrows():
                start_sample = int(row["start_time"] * sr)
                end_sample = int(row["stop_time"] * sr)
                audio_chunks.append(y[start_sample:end_sample])

            if audio_chunks:
                concatenated = np.concatenate(audio_chunks)
                output_path = os.path.join(output_dir, f"{singer_id}.wav")
                sf.write(output_path, concatenated, sr)
                output_paths.append(output_path)
                logger.info("Saved speaker audio: %s", output_path)

        return output_paths


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Speaker diarization with custom embeddings"
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to audio file or directory",
    )
    parser.add_argument(
        "--min_seg_dur",
        type=float,
        default=0.5,
        help="Minimum segment duration in seconds",
    )
    parser.add_argument(
        "--custom_model_path",
        type=str,
        required=True,
        help="Path to custom RawNet3 model checkpoint",
    )
    return parser.parse_args()


def main():
    """Main entry point for CLI usage."""
    args = parse_args()
    ds_path = args.ds_path

    # Build file list
    if os.path.isdir(ds_path):
        file_list = [
            os.path.join(ds_path, f)
            for f in os.listdir(ds_path)
            if f.endswith(".wav")
            and not f.startswith(".")
            and not os.path.isdir(os.path.join(ds_path, f))
        ]
    elif os.path.isfile(ds_path):
        file_list = [ds_path]
        ds_path = os.path.dirname(ds_path)
    else:
        raise ValueError(f"Invalid dataset path: {ds_path}")

    # Setup output directories
    diar_csvs_dir = os.path.join(ds_path, "single_singer_diarizations")
    os.makedirs(diar_csvs_dir, exist_ok=True)
    audio_out_dir = os.path.join(".", "singer_extracts")
    os.makedirs(audio_out_dir, exist_ok=True)

    # Initialize diarizer once
    diarizer = Diarizer(custom_model_path=args.custom_model_path)

    # Process each file
    for audio_path in file_list:
        track_name = os.path.splitext(os.path.basename(audio_path))[0]
        dst_csv_path = os.path.join(diar_csvs_dir, f"{track_name}.csv")

        # Run diarization
        segments_df = diarizer.diarize(
            audio_path,
            min_seg_dur=args.min_seg_dur,
            verbose=True,
        )

        # Save segments CSV
        segments_df.to_csv(dst_csv_path, index=False)
        logger.info("Saved segments to: %s", dst_csv_path)

        # Extract speaker audio
        diarizer.extract_speaker_audio(
            audio_path,
            segments_df,
            audio_out_dir,
        )


if __name__ == "__main__":
    main()
