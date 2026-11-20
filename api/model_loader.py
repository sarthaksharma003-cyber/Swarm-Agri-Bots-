"""Model lifecycle manager ensuring YOLO weights are loaded once at startup."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from api.config import (
    DEFAULT_MODEL_PATH,
    FALLBACK_MODEL_PATH,
    MODEL_PATH,
    VERIFIED_CLASS_MAPPING,
)

logger = logging.getLogger("SwarmAgriAPI.ModelLoader")


class ModelManager:
    """Singleton manager for the YOLO object detection model lifecycle."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.is_loaded: bool = False
        self.model_path: str = ""
        self.classes: Dict[int, str] = VERIFIED_CLASS_MAPPING.copy()
        self.device: str = "cpu"
        self.load_error: Optional[str] = None

    def load_model(self, target_path: Optional[Path] = None) -> bool:
        """Load the YOLO model checkpoint once into memory.
        
        Searches the configured path first, then falls back to fallback path
        if necessary. Does not reload if already loaded with the same path.
        """
        # Determine target checkpoint path
        candidate = target_path or MODEL_PATH
        if not candidate.exists() and candidate != FALLBACK_MODEL_PATH:
            if FALLBACK_MODEL_PATH.exists():
                logger.warning(
                    f"Configured model path {candidate} not found. "
                    f"Falling back to {FALLBACK_MODEL_PATH}"
                )
                candidate = FALLBACK_MODEL_PATH

        self.model_path = str(candidate)

        if not candidate.exists():
            self.is_loaded = False
            self.load_error = f"Model checkpoint not found at {candidate}"
            logger.error(self.load_error)
            return False

        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model from {candidate} ...")
            self.model = YOLO(str(candidate))
            
            # Detect available inference device
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # Extract and verify class names
            if hasattr(self.model, "names") and self.model.names:
                # Convert keys to int if necessary
                self.classes = {int(k): str(v) for k, v in self.model.names.items()}
            else:
                self.classes = VERIFIED_CLASS_MAPPING.copy()

            self.is_loaded = True
            self.load_error = None
            logger.info(
                f"Model successfully loaded on {self.device}. "
                f"Verified classes: {self.classes}"
            )
            return True
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            return False

    def set_mock_model(
        self,
        mock_model: Any,
        model_path: str = "mocked_yolo11n.pt",
        classes: Optional[Dict[int, str]] = None,
        device: str = "cpu",
    ) -> None:
        """Inject a mock model for lightweight unit and integration testing."""
        self.model = mock_model
        self.is_loaded = True
        self.model_path = model_path
        self.classes = classes or VERIFIED_CLASS_MAPPING.copy()
        self.device = device
        self.load_error = None
        logger.info("Injected mock model for testing.")

    def unload(self) -> None:
        """Unload model from memory on shutdown."""
        self.model = None
        self.is_loaded = False
        self.load_error = None
        logger.info("YOLO model unloaded.")


# Global singleton instance used across the FastAPI application
model_manager = ModelManager()
