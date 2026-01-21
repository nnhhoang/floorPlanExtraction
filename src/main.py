"""
PDF Floor Plan Extractor - Main CLI Interface

Extract floor plan images and grid line coordinates from Taisei BIM standard PDFs.
"""
import re
import click
import cv2
from pathlib import Path
from src.core.pdf_handler import PDFHandler
from src.core.grid_detector import GridDetector
from src.core.image_processor import ImageProcessor
from src.core.auto_detector import AutoDetector
from src.utils.logger import log


def sanitize_filename(name: str) -> str:
    """
    Remove potentially dangerous characters from filename.

    Args:
        name: Original filename

    Returns:
        Sanitized filename safe for filesystem operations
    """
    # Remove path separators and null bytes
    sanitized = re.sub(r'[/\\:\x00]', '_', name)
    # Keep only alphanumeric, underscore, hyphen, and period
    sanitized = re.sub(r'[^\w\-.]', '_', sanitized)
    # Prevent directory traversal
    sanitized = sanitized.lstrip('.')
    return sanitized if sanitized else 'unnamed'


@click.command()
@click.option(
    '--pdf',
    '-p',
    type=click.Path(exists=True),
    required=True,
    help='Path to input PDF file'
)
@click.option(
    '--page',
    '-n',
    type=int,
    required=True,
    help='Page number to extract (1-indexed)'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='output',
    help='Output directory for results (default: output/)'
)
@click.option(
    '--dpi',
    '-d',
    type=click.IntRange(72, 1200),
    default=300,
    help='Resolution in DPI for image extraction (72-1200, default: 300)'
)
@click.option(
    '--visualize',
    '-v',
    is_flag=True,
    help='Save visualization of detected grid lines'
)
@click.option(
    '--use-gpu',
    is_flag=True,
    help='Use GPU for OCR processing (requires CUDA)'
)
@click.option(
    '--min-line-ratio',
    type=float,
    default=0.6,
    help='Minimum line length ratio (0.0-1.0) for grid line filtering (default: 0.6 = 60%%)'
)
@click.option(
    '--min-area-ratio',
    type=float,
    default=0.15,
    help='Minimum area ratio for floor plan detection (default: 0.15 = 15%%)'
)
@click.option(
    '--no-auto-crop',
    is_flag=True,
    help='Disable automatic floor plan region detection and cropping'
)
@click.option(
    '--auto-detect',
    '-a',
    is_flag=True,
    help='Auto-detect grid label configuration (circles, format, prefixes)'
)
@click.option(
    '--multi-char/--single-char',
    default=True,
    help='Label format: multi-char (X1/Y1) or single-char (1/A)'
)
@click.option(
    '--has-circle/--no-circle',
    default=True,
    help='Whether labels are in circles'
)
@click.option(
    '--longitude-prefix',
    default='X',
    help="Prefix for vertical axis labels (default: 'X', use '' for numbers)"
)
@click.option(
    '--latitude-prefix',
    default='Y',
    help="Prefix for horizontal axis labels (default: 'Y', use '' for letters)"
)
def extract_floor_plan(pdf, page, output_dir, dpi, visualize, use_gpu, min_line_ratio, 
                       min_area_ratio, no_auto_crop, auto_detect, multi_char, has_circle,
                       longitude_prefix, latitude_prefix):
    """
    Extract floor plan image and grid line coordinates from PDF.

    This tool processes Taisei BIM standard PDFs and:
    1. Extracts the specified page as a high-resolution PNG image
    2. Detects grid lines (X and Y axes) and their coordinates
    3. Exports grid data to CSV format

    Example usage:
        python -m src.main -p input.pdf -n 5 -o results/
    """
    try:
        log.info("=" * 60)
        log.info("PDF Floor Plan Extractor")
        log.info("=" * 60)

        pdf_path = Path(pdf)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filenames with sanitization for security
        base_name = sanitize_filename(f"{pdf_path.stem}_page{page}")
        png_path = output_dir / f"{base_name}.png"
        cropped_path = output_dir / f"{base_name}_cropped.png"
        csv_path = output_dir / f"{base_name}_gridlines.csv"
        vis_path = output_dir / f"{base_name}_visualization.png"
        rect_vis_path = output_dir / f"{base_name}_detected_region.png"

        # Step 1: Extract PDF page as PNG
        log.info(f"\n[1/4] Extracting page {page} from PDF...")
        log.info(f"      Input: {pdf_path}")

        with PDFHandler(pdf_path) as pdf_handler:
            # Check page count
            total_pages = pdf_handler.get_page_count()
            log.info(f"      Total pages: {total_pages}")

            if page < 1 or page > total_pages:
                raise click.BadParameter(
                    f"Page {page} is out of range. PDF has {total_pages} pages."
                )

            # Extract and save page as PNG
            pdf_handler.save_page_as_png(page, png_path, dpi=dpi)
            log.info(f"      ✓ Full page saved: {png_path}")

            # Load image for grid detection
            image = cv2.imread(str(png_path))

        # Step 2: Detect and crop floor plan region FIRST
        log.info(f"\n[2/5] Detecting floor plan region...")

        # Pre-crop the floor plan before auto-detection (same as test.py)
        cropped_image = image
        crop_rect = None
        if not no_auto_crop:
            cropped_image, crop_rect = ImageProcessor.find_and_crop_floor_plan(
                image,
                min_area_ratio=min_area_ratio,
                padding=10
            )
            if crop_rect:
                x, y, w, h = crop_rect
                log.info(f"      ✓ Floor plan region detected: {w}x{h} at ({x}, {y})")
                
                # Save cropped image
                cv2.imwrite(str(cropped_path), cropped_image)
                log.info(f"      ✓ Cropped floor plan saved: {cropped_path}")
            else:
                log.warning("      ⚠ No floor plan region detected, using full page")
                cropped_image = image

        detector = GridDetector(use_gpu=use_gpu)

        # Step 3: Auto-detect or use manual configuration
        # Run on CROPPED image (same as test.py)
        if auto_detect:
            log.info(f"\n[3/5] Auto-detecting grid configuration...")
            auto_detector = AutoDetector(use_gpu=use_gpu)
            config = auto_detector.detect_configuration(cropped_image)  # Use cropped image!
            
            # Use detected configuration
            is_multi_characters = config.is_multi_characters
            has_circle_detected = config.has_circle
            long_prefix = config.longitude_prefix
            lat_prefix = config.latitude_prefix
            
            log.info(f"      ✓ Auto-detected: {config}")
        else:
            # Use manual configuration
            is_multi_characters = multi_char
            has_circle_detected = has_circle
            long_prefix = longitude_prefix
            lat_prefix = latitude_prefix
            
            log.info(f"\n[3/5] Using manual configuration...")
            log.info(f"      Format: {'Multi-char' if is_multi_characters else 'Single-char'}")
            log.info(f"      Circles: {has_circle_detected}")
            log.info(f"      Prefixes: longitude='{long_prefix}', latitude='{lat_prefix}'")

        # Step 4: Detect grid lines on cropped image
        log.info(f"\n[4/5] Detecting grid lines...")

        # Log detection parameters
        log.info(f"      Detection settings:")
        log.info(f"        - Min line ratio: {min_line_ratio:.1%}")
        log.info(f"        - Min area ratio: {min_area_ratio:.1%}")

        grid_lines, _ = detector.process_floor_plan(
            cropped_image,  # Use cropped image
            is_multi_characters=is_multi_characters,
            has_circle=has_circle_detected,
            longitude_prefix=long_prefix,
            latitude_prefix=lat_prefix,
            auto_crop=False,  # Already cropped!
            min_area_ratio=min_area_ratio,
            min_line_length_ratio=min_line_ratio
        )

        # Optionally save visualization of detected rectangle
        if visualize and crop_rect:
            rect_vis = ImageProcessor.visualize_rectangle(image, crop_rect)
            cv2.imwrite(str(rect_vis_path), rect_vis)
            log.info(f"      ✓ Region detection visualization: {rect_vis_path}")

        # Step 5: Grid line detection results
        log.info(f"\n[5/6] Grid line detection results...")

        if not grid_lines:
            log.warning("      ⚠ No grid lines detected!")
            log.info("\nProcess completed with warnings.")
            return

        log.info(f"      ✓ Detected {len(grid_lines)} grid lines")

        # Count X and Y lines
        x_lines = sum(1 for line in grid_lines if line.is_vertical)
        y_lines = sum(1 for line in grid_lines if not line.is_vertical)
        log.info(f"        - X-axis (vertical): {x_lines} lines")
        log.info(f"        - Y-axis (horizontal): {y_lines} lines")

        # Step 6: Export to CSV
        log.info(f"\n[6/6] Exporting grid data...")

        detector.export_to_csv(grid_lines, csv_path)
        log.info(f"      ✓ Grid data saved: {csv_path}")

        # Optional: Save visualization on cropped image
        if visualize:
            detector.visualize_grid_lines(cropped_image, grid_lines, vis_path)
            log.info(f"      ✓ Grid lines visualization: {vis_path}")

        # Summary
        log.info("\n" + "=" * 60)
        log.info("✓ Process completed successfully!")
        log.info("=" * 60)
        log.info(f"Output files:")
        log.info(f"  - Full page PNG:     {png_path}")
        if crop_rect:
            log.info(f"  - Cropped floor plan: {cropped_path}")
        log.info(f"  - Grid data CSV:      {csv_path}")
        if visualize:
            log.info(f"  - Grid visualization: {vis_path}")
            if crop_rect:
                log.info(f"  - Region detection:   {rect_vis_path}")
        log.info("=" * 60)

    except Exception as e:
        log.error(f"\n✗ Error: {e}")
        raise click.ClickException(str(e))


@click.group()
def cli():
    """PDF Floor Plan Extractor - CLI Tool"""
    pass


cli.add_command(extract_floor_plan)


if __name__ == '__main__':
    extract_floor_plan()
