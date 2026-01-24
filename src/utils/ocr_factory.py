"""OCR Factory for shared PaddleOCR instance management.

PaddleOCR with PP-OCRv4/v5 mobile model is faster and lighter than EasyOCR.
This factory ensures only one instance is created and shared across
GridDetector and AutoDetector classes.
"""
from paddleocr import PaddleOCR
from typing import Optional, List, Tuple, Any
import numpy as np
import os
from src.utils.logger import log


class PaddleOCRWrapper:
    """Wrapper to provide EasyOCR-compatible interface for PaddleOCR."""

    def __init__(self, use_gpu: bool = False):
        """Initialize PaddleOCR with PP-OCRv4 mobile model."""
        # Suppress model download messages
        os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

        # PaddleOCR 3.x uses simplified API
        # Use PP-OCRv4 with mobile models for speed
        self._ocr = PaddleOCR(
            lang='en',
            ocr_version='PP-OCRv4',  # Use v4 mobile models
            use_doc_orientation_classify=False,  # Disable for speed
            use_doc_unwarping=False,  # Disable for speed
            use_textline_orientation=False,  # Disable for speed
        )

    def readtext(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """
        Read text from image with EasyOCR-compatible output format.

        Args:
            image: Input image as numpy array (BGR or grayscale)

        Returns:
            List of (bbox, text, confidence) tuples where:
            - bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] corner points
            - text: Recognized text string
            - confidence: Recognition confidence (0-1)
        """
        # PaddleOCR expects RGB, convert if grayscale
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = np.stack([image, image, image], axis=-1)

        # Run OCR using predict() method (PaddleOCR 3.x API)
        results = self._ocr.predict(image)

        # Convert to EasyOCR format
        output = []
        if results:
            for result in results:
                # Each result is a dict with rec_texts, rec_scores, rec_polys
                texts = result.get('rec_texts', [])
                scores = result.get('rec_scores', [])
                polys = result.get('rec_polys', [])

                for i, (text, score) in enumerate(zip(texts, scores)):
                    if i < len(polys):
                        bbox = polys[i].tolist()
                    else:
                        bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]
                    output.append((bbox, text, score))

        return output


class OCRFactory:
    """Factory class for managing shared PaddleOCR reader instance."""

    _instance: Optional[PaddleOCRWrapper] = None
    _use_gpu: bool = False

    @classmethod
    def get_reader(cls, use_gpu: bool = False) -> PaddleOCRWrapper:
        """
        Get or create a shared PaddleOCR reader instance.

        Args:
            use_gpu: Use GPU for OCR if available (default: False).
                     Note: Once created, the GPU setting cannot be changed.

        Returns:
            Shared PaddleOCR Wrapper instance with EasyOCR-compatible interface
        """
        if cls._instance is None:
            cls._use_gpu = use_gpu
            try:
                cls._instance = PaddleOCRWrapper(use_gpu=use_gpu)
                gpu_status = 'GPU' if use_gpu else 'CPU'
                log.info(f"OCRFactory: Created shared PaddleOCR instance ({gpu_status})")
            except Exception as e:
                log.error(f"OCRFactory: Failed to create PaddleOCR instance: {e}")
                raise
        elif cls._use_gpu != use_gpu:
            log.warning(
                f"OCRFactory: Requested GPU={use_gpu} but instance already created "
                f"with GPU={cls._use_gpu}. Using existing instance."
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the shared instance (useful for testing).

        Note: This doesn't free GPU memory until garbage collected.
        """
        cls._instance = None
        log.debug("OCRFactory: Instance reset")
