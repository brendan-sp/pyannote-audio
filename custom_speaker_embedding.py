import sys
import numpy as np
import torch
import torch.nn.functional as F
from functools import cached_property
from torch.nn.utils.rnn import pad_sequence
from pyannote.audio.core.inference import BaseInference
sys.path.insert(0, "/home/brendanoconnor/sc-hiro-backend")
sys.path.insert(0, "/home/brendanoconnor/sc-hiro-backend/voice_bio_service")
from espnet2.bin.spk_inference import Speech2Embedding
# Now voice_bio is a top-level module, matching the internal imports
from voice_bio.rawnet3 import RawNet3Inferencer

class MyRawNet3Inferencer(RawNet3Inferencer):
    def __init__(self):
        """Initialize RawNet3 model and VAD."""
        # Initialize device for model placement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = str(self.device)  # Convert to string for ESPnet

        # Download model checkpoint
        model_checkpoint_path = "/home/brendanoconnor/.cache/voice-biometrics/train_12m/58epoch_eer=7.96.pth"

        # Initialize ESPnet model with correct device
        pretrained_model = Speech2Embedding.from_pretrained(
            model_tag="espnet/voxcelebs12_rawnet3",
            device=device_str,  # Pass device to ESPnet!
        )

        # Attempting to hot swap the weights, avoiding missing config file.
        fine_tuned_weights = torch.load(model_checkpoint_path, map_location=self.device)

        # Remove the loss.weight and loss.bias if they exist in the state dict
        if "loss.weight" in fine_tuned_weights:
            del fine_tuned_weights["loss.weight"]
        if "loss.bias" in fine_tuned_weights:
            del fine_tuned_weights["loss.bias"]

        pretrained_model.spk_model.load_state_dict(fine_tuned_weights, strict=False)
        self.model = pretrained_model
        # Note: ESPnet handles device placement internally via the device parameter

        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad:v4.0",
            model="silero_vad",
        )

        # Move VAD model to device (fixes CPU bottleneck)
        self.vad_model = self.vad_model.to(self.device)
    
    @cached_property
    def sample_rate(self) -> int:
        """Expected audio sample rate"""
        return 16000
    
    @cached_property
    def dimension(self) -> int:
        """Embedding dimension"""
        return self.model.spk_train_args.projector_conf['output_size']
    
    @cached_property
    def min_num_samples(self) -> int:
        """Minimum number of audio samples needed for valid embedding"""
        # ~0.5 seconds at 16kHz
        return 8000

    def to(self, device: torch.device) -> "MyRawNet3Inferencer":
        """Move model to device"""
        self.device = device
        # ESPnet model handles its own device placement
        # VAD model needs explicit move
        self.vad_model = self.vad_model.to(device)
        return self

    def __call__(
        self, 
        waveforms: torch.Tensor, 
        masks: torch.Tensor = None
    ) -> np.ndarray:
        """Extract embeddings from waveforms (pyannote-compatible interface)
        
        Parameters
        ----------
        waveforms : torch.Tensor
            Shape: (batch_size, num_channels, num_samples)
            Audio waveforms (num_channels is typically 1 for mono)
        masks : torch.Tensor, optional
            Shape: (batch_size, num_frames)
            Binary masks indicating which frames to use for each speaker
            
        Returns
        -------
        embeddings : np.ndarray
            Shape: (batch_size, dimension)
            Speaker embeddings
        """
        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1, "Only mono audio supported"
        
        # Remove channel dimension: (batch, 1, samples) -> (batch, samples)
        waveforms = waveforms.squeeze(dim=1)
        
        if masks is None:
            # No mask - use entire waveform for each sample in batch
            with torch.inference_mode():
                embeddings = self._extract_batch_embeddings(waveforms)
            return embeddings
        
        # Handle masked extraction (for speaker-specific regions)
        batch_size_masks, num_frames = masks.shape
        assert batch_size == batch_size_masks
        
        # Interpolate mask from frame-level to sample-level
        imasks = F.interpolate(
            masks.unsqueeze(dim=1), 
            size=num_samples, 
            mode="nearest"
        ).squeeze(dim=1) > 0.5
        
        # Extract masked segments and pad to equal length
        signals = pad_sequence(
            [waveform[imask] for waveform, imask in zip(waveforms, imasks)],
            batch_first=True,
        )
        
        wav_lens = imasks.sum(dim=1)
        max_len = wav_lens.max()
        
        # Corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.nan * np.zeros((batch_size, self.dimension))
        
        too_short = wav_lens < self.min_num_samples
        
        with torch.inference_mode():
            embeddings = self._extract_batch_embeddings(signals)
        
        # Mark too-short segments as NaN
        embeddings[too_short.cpu().numpy()] = np.nan
        
        return embeddings
    
    def _extract_batch_embeddings(self, waveforms: torch.Tensor) -> np.ndarray:
        """Extract embeddings for a batch of waveforms
        
        Parameters
        ----------
        waveforms : torch.Tensor
            Shape: (batch_size, num_samples)
            
        Returns
        -------
        embeddings : np.ndarray
            Shape: (batch_size, dimension)
        """
        batch_size = waveforms.shape[0]
        embeddings_list = []
        
        for i in range(batch_size):
            waveform = waveforms[i:i+1]  # Keep batch dim: (1, num_samples)
            
            # ESPnet Speech2Embedding returns a generator/list of embeddings
            emb_result = self.model(waveform.squeeze(0).to(self.device))
            
            # Average embeddings if multiple are returned
            emb_tensor = torch.mean(torch.stack(tuple(emb_result)), dim=0)
            embeddings_list.append(emb_tensor)
        
        # Stack all embeddings: (batch_size, dimension)
        embeddings = torch.stack(embeddings_list, dim=0)
        return embeddings.cpu().numpy()


class MyCustomSpeakerEmbedding(BaseInference):
    """Custom speaker embedding wrapper for pyannote.audio
    
    This is a thin wrapper around MyRawNet3Inferencer that conforms to
    pyannote's BaseInference interface.
    
    Parameters
    ----------
    model_path : str
        Path to your custom model checkpoint (currently unused, model path is hardcoded)
    device : torch.device, optional
        Device to run inference on
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the RawNet3 model
        self.model = MyRawNet3Inferencer()
        
        if device is not None:
            self.model.to(device)
    
    def to(self, device: torch.device) -> "MyCustomSpeakerEmbedding":
        """Move model to device"""
        self.model.to(device)
        self.device = device
        return self
    
    @cached_property
    def sample_rate(self) -> int:
        """Expected audio sample rate"""
        return self.model.sample_rate
    
    @cached_property
    def dimension(self) -> int:
        """Embedding dimension"""
        return self.model.dimension
    
    @cached_property
    def metric(self) -> str:
        """Distance metric for comparing embeddings"""
        return "cosine"
    
    @cached_property
    def min_num_samples(self) -> int:
        """Minimum number of audio samples needed for valid embedding"""
        return self.model.min_num_samples
    
    def __call__(
        self, 
        waveforms: torch.Tensor, 
        masks: torch.Tensor = None
    ) -> np.ndarray:
        """Extract embeddings from waveforms
        
        Parameters
        ----------
        waveforms : torch.Tensor
            Shape: (batch_size, num_channels, num_samples)
            Audio waveforms (num_channels is typically 1 for mono)
        masks : torch.Tensor, optional
            Shape: (batch_size, num_frames)
            Binary masks indicating which frames to use
            
        Returns
        -------
        embeddings : np.ndarray
            Shape: (batch_size, dimension)
            Speaker embeddings
        """
        # Delegate to the model's __call__ which handles all the logic
        return self.model(waveforms, masks)