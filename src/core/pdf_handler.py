"""PDF handling and page extraction module"""
import pymupdf as fitz  # PyMuPDF (newer version uses pymupdf)
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from src.utils.logger import log


class PDFHandler:
    """Handle PDF processing and page extraction."""

    def __init__(self, pdf_path: str | Path):
        """
        Initialize PDF handler.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        self.pdf_document: Optional[fitz.Document] = None
        self._load_pdf()

    def _load_pdf(self):
        """Load PDF document."""
        try:
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

            self.pdf_document = fitz.open(str(self.pdf_path))
            log.info(f"PDF loaded successfully: {self.pdf_document.page_count} pages")

        except Exception as e:
            log.error(f"Failed to load PDF: {e}")
            raise

    def get_page_count(self) -> int:
        """
        Get total number of pages in PDF.

        Returns:
            Number of pages
        """
        if not self.pdf_document:
            return 0
        return self.pdf_document.page_count

    def extract_page_as_image(
        self,
        page_number: int,
        dpi: int = 300,
        output_format: str = "PIL"
    ) -> Image.Image | np.ndarray:
        """
        Extract a specific page as high-resolution image.

        Args:
            page_number: Page number (1-indexed for user, converted to 0-indexed)
            dpi: Resolution in DPI (default: 300 for high quality)
            output_format: Output format ("PIL" or "numpy")

        Returns:
            PIL Image or numpy array

        Raises:
            ValueError: If page number is invalid
        """
        if not self.pdf_document:
            raise ValueError("PDF document not loaded")

        # Convert to 0-indexed
        page_idx = page_number - 1

        if page_idx < 0 or page_idx >= self.pdf_document.page_count:
            raise ValueError(
                f"Invalid page number: {page_number}. "
                f"PDF has {self.pdf_document.page_count} pages"
            )

        try:
            page = self.pdf_document[page_idx]

            # Calculate scaling matrix for desired DPI
            # PDF default is 72 DPI, so we scale accordingly
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            # Render page to pixmap (high quality)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Get image size
            width = pix.width
            height = pix.height

            log.info(
                f"Extracted page {page_number} at {dpi} DPI "
                f"(size: {width}x{height} pixels)"
            )

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if output_format.lower() == "numpy":
                return np.array(img)
            return img

        except Exception as e:
            log.error(f"Failed to extract page {page_number}: {e}")
            raise

    def save_page_as_png(
        self,
        page_number: int,
        output_path: str | Path,
        dpi: int = 300
    ) -> Path:
        """
        Extract and save a specific page as PNG file.

        Args:
            page_number: Page number (1-indexed)
            output_path: Output file path for PNG
            dpi: Resolution in DPI

        Returns:
            Path to saved PNG file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract page as PIL Image
        img = self.extract_page_as_image(page_number, dpi=dpi, output_format="PIL")

        # Save as PNG
        img.save(output_path, "PNG", optimize=True)

        log.info(f"Saved page {page_number} to: {output_path}")
        return output_path

    def get_metadata(self) -> dict:
        """
        Get PDF metadata.

        Returns:
            Dictionary containing PDF metadata
        """
        if not self.pdf_document:
            return {}

        metadata = self.pdf_document.metadata
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "page_count": self.pdf_document.page_count,
            "is_encrypted": self.pdf_document.is_encrypted,
        }

    def close(self):
        """Close PDF document."""
        if self.pdf_document:
            self.pdf_document.close()
            log.debug("PDF document closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
