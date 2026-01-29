## Project Overview

PDF Floor Plan Extractor - A Python tool that extracts floor plan images and grid line coordinates from T-Company BIM standard PDFs. Designed to run both locally and on AWS Lambda.

## Commands

### Run CLI
```bash
python -m src.main -p <pdf_path> -n <page_number> -o <output_dir>
```

Key CLI options:
- `-p, --pdf`: Input PDF file path
- `-n, --page`: Page number (1-indexed)
- `-o, --output-dir`: Output directory (default: `output/`)
- `-d, --dpi`: Resolution for image extraction (default: 300)
- `-v, --visualize`: Save grid line visualization images
- `-a, --auto-detect`: Auto-detect grid label configuration
- `--use-gpu`: Enable GPU for OCR (requires CUDA)
- `--min-line-ratio`: Grid line filtering threshold (default: 0.6)
- `--no-auto-crop`: Disable automatic floor plan region detection

### Run Tests
```bash
# Run all tests
python test.py

# Run specific test
python test.py --test <test_name>

# List available tests
python test.py --list

# Test with custom PDF
python test.py --pdf <path_to_pdf>
```

Test output goes to `test/output/`. Test input PDFs should be placed in `test/input/`.

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Core Processing Pipeline

1. **PDFHandler** (`src/core/pdf_handler.py`) - Extracts PDF pages as high-resolution PNG images using PyMuPDF
2. **ImageProcessor** (`src/core/image_processor.py`) - Detects and crops the floor plan region (largest rectangle with black border)
3. **AutoDetector** (`src/core/auto_detector.py`) - Auto-detects grid label configuration (circles, format, prefixes)
4. **GridDetector** (`src/core/grid_detector.py`) - Detects grid lines and extracts coordinates using OCR (PaddleOCR)

### Grid Label Formats

The system supports two grid label formats:
- **Multi-character**: `X1, X2, Y1, Y2` (prefixes like X/Y or M/N)
- **Single-character**: `1, 2, A, B` (pure numbers for longitude, letters for latitude)

Labels can be enclosed in circles or displayed as plain text.

### Key Data Structures

- `GridLine` dataclass: Represents a detected grid line with label, start/end coordinates, angle, and orientation
- `AutoConfig` dataclass: Holds auto-detected configuration (has_circle, is_multi_characters, prefixes)
- `GridLabelConfig` dataclass (in test.py): Test configuration for grid label detection
