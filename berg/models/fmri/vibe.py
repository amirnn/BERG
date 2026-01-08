import os
import numpy as np
import torch
import yaml
from torchvision import transforms as trn
from typing import Dict, Any, Optional
from berg.interfaces.base_model import BaseModelInterface
from berg.core.model_registry import register_model
from berg.core.exceptions import ModelLoadError, InvalidParameterError, StimulusError
from berg.core.parameter_validator import (
    validate_subject,
    validate_selection_keys,
    validate_roi,
)

from dataclasses import dataclass

@dataclass
class Config:
    num_subjects: int
    features_extractor_llm: str
    d_model: int
    depth: int


# Load model info from YAML
def load_model_info():
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-vibe.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        return yaml.safe_load(f)


# Load model_info once at the top
model_info = load_model_info()

# Register this model with the registry using model_info
register_model(
    model_id=model_info["vibe"],
    module_path="berg.models.fmri.vibe",  # Replace with actual path
    class_name="VIBE", #TODO
    modality=model_info.get("modality", "fmri"),
    training_dataset=model_info.get("training_dataset", "your_dataset"), #TODO
    yaml_path=os.path.join(os.path.dirname(__file__), "..", "model_cards", "fmri-vibe.yaml")
)

#TODO: Improve the name
#TODO: Implement the class methods
class VIBE(BaseModelInterface):
    """
    Your model description here. Explain what this model does, what
    in silico neural responses it generates, and any other important details.
    """

    MODEL_ID = model_info["model_id"]
    # Extract any validation info from model_info
    VALID_SUBJECTS = model_info["parameters"]["subject"]["valid_values"]
    # Supported Configs
    PRETRAINED_CONFIGS: list[Config] = [
        Config(num_subjects=10, features_extractor_llm="Qwen-0.5B", d_model=32, depth=12),
        Config(num_subjects=10, features_extractor_llm="Qwen-1.5B", d_model=32, depth=12),
        Config(num_subjects=10, features_extractor_llm="Qwen-3.5B", d_model=32, depth=12)
    ]

    def __init__(self, config: Config,
                 selection: Dict, device: str = "auto", berg_dir: Optional[str] = None, **kwargs):
        """
        Initialize your model with the required parameters.

        Parameters
        ----------
        subject : int
            Subject ID for subject-specific models.
        selection : dict
            Specifies which outputs to include in the model responses
            (ROI, Time interval, ...)
        device : str
            Device to run the model on ('cpu', 'cuda', or 'auto').
        berg_dir : str, optional
            Path to the BERG directory.
        **kwargs
            Additional model-specific parameters.
        """
        self.subject = subject
        self.berg_dir = berg_dir
        self.model = None
        self.selection = selection
        self._validate_parameters()

        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Store any additional parameters
        # self.your_param = kwargs.get('your_param', default_value)

    def _validate_parameters(self):
        """
        Validate the input parameters against the model specs.
        """
        if self.subject not in self.VALID_SUBJECTS:
            raise InvalidParameterError(
                f"Subject must be one of {self.VALID_SUBJECTS}, got {self.subject}"
            )

        # For selection Paramter if available
        if self.selection is not None:
            # Validate selection keys
            validate_selection_keys(self.selection, self.SELECTION_KEYS)

            # Individual validations (example of ROIs)
            if "roi" in self.selection:
                self.roi = validate_roi(
                    self.selection["roi"], self.VALID_ROIS
                )
        # Ensure selection is provided
        else:
            raise InvalidParameterError("Parameter 'selection' is required but was not provided")

        # Add any other parameter validation here
    
    def load_model(self) -> None:
        """
        Load model weights and prepare for inference.
        """
        try:
            # Build paths to model weights
            weights_path = os.path.join(
                self.berg_dir,
                'your_path')  # Adjust filename format as needed

            # Load your model here
            # Example with PyTorch:
            # self.model = YourModelArchitecture()
            # self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
            # self.model.to(self.device)
            # self.model.eval()

            print(f"Model loaded on {self.device} for subject {self.subject}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    def generate_response(
    self,
    stimulus: np.ndarray,
    **kwargs) -> np.ndarray:
        """
        Generate in silico neural responses for given stimuli.

        Parameters
        ----------
        stimulus : np.ndarray
            Input stimulus array. Typically has shape (batch_size, channels, height, width)
            for image stimuli, but requirements vary by model.
        **kwargs
            Additional model-specific parameters for in silico response generation.

        Returns
        -------
        np.ndarray
            Simulated in silico neural responses. Shape depends on your model's output.
        """
        # Validate stimulus
        if not isinstance(stimulus, np.ndarray) or len(stimulus.shape) != 4:
            raise StimulusError(
                "Stimulus must be a 4D numpy array (batch, channels, height, width)"
            )

        # Preprocess stimulus if needed
        preprocessed_stimulus = preprocess(stimulus)

        # Generate in silico responses
        with torch.no_grad():
            batch_size = 100  # Adjust as needed
            responses = []

            for i in range(0, len(stimulus), batch_size):
                batch = torch.from_numpy(stimulus[i:i+batch_size]).to(self.device)
                output = self.model(batch)
                responses.append(output.cpu().numpy())

            all_responses = np.concatenate(responses, axis=0)

        return all_responses
    
    
    @classmethod
    def get_metadata(cls, berg_dir=None, subject=None, model_instance=None, roi=None, **kwargs) -> Dict[str, Any]:
        """
        Retrieve metadata for the model.

        Parameters
        ----------
        berg_dir : str
            Path to the BERG directory where metadata is stored.
        subject : int
            Subject number.
        model_instance : BaseModelInterface, optional
            If provided, parameters can be extracted directly from the model instance.
        roi : str, optional
            Region of interest (if applicable).
        **kwargs
            Additional model-specific parameters.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        """

        # Extract parameters from instance if available
        if model_instance is not None:
            berg_dir = model_instance.berg_dir
            subject = model_instance.subject
            roi = getattr(model_instance, "roi", roi)

        # Also allow metadata retrieval from class instance
        elif not isinstance(cls, type) and isinstance(cls, BaseModelInterface):
            berg_dir = cls.berg_dir
            subject = cls.subject
            roi = getattr(cls, "roi", roi)

        # Validate required parameters
        missing = []
        if berg_dir is None: missing.append("berg_dir")
        if subject is None: missing.append("subject")
        if roi is None and "VALID_ROIS" in dir(cls): missing.append("roi")

        if missing:
            raise InvalidParameterError(f"Required parameters missing: {', '.join(missing)}")

        # Optional: validate against allowed values
        validate_subject(subject, cls.VALID_SUBJECTS)
        if roi is not None and hasattr(cls, "VALID_ROIS"):
            validate_roi(roi, cls.VALID_ROIS)

        # Build metadata path
        filename = os.path.join(
            berg_dir,
            "encoding_models",
            "modality-<your_modality>",               # e.g., modality-fmri
            "train_dataset-<your_dataset>",           # e.g., train_dataset-nsd
            "model-<your_model_id>",                  # e.g., model-vit_b_32
            "metadata",
            f"metadata_sub-{subject:02d}" + (f"_roi-{roi}" if roi else "") + ".npy"
        )

        # Load metadata
        if os.path.exists(filename):
            metadata = np.load(filename, allow_pickle=True).item()
            return metadata
        else:
            raise FileNotFoundError(f"Metadata file not found at: {filename}")

    @classmethod
    def get_model_id(cls) -> str:
        """
        Return the model's unique identifier.

        Returns
        -------
        str
            Model ID string from the YAML config.
        """
        return cls.MODEL_ID

    def cleanup(self) -> None:
        """
        Release resources (e.g., GPU memory) when finished.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Free GPU memory if using CUDA
            if hasattr(self.model, 'to'):
                self.model.to('cpu')

            # Clear references
            self.model = None

            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()