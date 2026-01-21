"""OCR Factory for shared EasyOCR instance management.

EasyOCR model loading is expensive (~2-5 seconds, ~500MB+ memory).
This factory ensures only one instance is created and shared across
GridDetector and AutoDetector classes.
"""
import easyocr
from typing import Optional
from src.utils.logger import log


class OCRFactory:
    """Factory class for managing shared EasyOCR reader instance."""

    _instance: Optional[easyocr.Reader] = None
    _use_gpu: bool = True

    @classmethod
    def get_reader(cls, use_gpu: bool = True) -> easyocr.Reader:
        """
        Get or create a shared EasyOCR reader instance.

        Args:
            use_gpu: Use GPU for OCR if available (default: True).
                     Note: Once created, the GPU setting cannot be changed.

        Returns:
            Shared EasyOCR Reader instance
        """
        if cls._instance is None:
            cls._use_gpu = use_gpu
            try:
                cls._instance = easyocr.Reader(['en'], gpu=use_gpu)
                gpu_status = 'GPU' if use_gpu else 'CPU'
                log.info(f"OCRFactory: Created shared EasyOCR instance ({gpu_status})")
            except Exception as e:
                log.error(f"OCRFactory: Failed to create EasyOCR instance: {e}")
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
