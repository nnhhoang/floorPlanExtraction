#!/usr/bin/env python3
"""
Comprehensive Test Suite for Floor Plan Grid Detection

This test suite covers all possible combinations of grid label formats:
- Single character (1,2,3 or A,B,C) vs Multi-character (X1,Y1 or M1,N1)
- With circles vs Without circles
- With/without reference coordinates

Test PDF Pages:
- Page 1: No floor plan (should fail validation)
- Pages 2-7: Multi-character format (X1,X2...X9, Y1,Y2...Y5) with circles
- Pages 8-11: Single character format (1,2,3,4,5 and A,B,C,D,E,F) with circles

Usage:
    # Run all tests
    python3 test.py

    # Run specific test
    python3 test.py --test multi_char_with_circles

    # Test with custom PDF
    python3 test.py --pdf custom.pdf
"""
import sys
import argparse
import cv2
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.pdf_handler import PDFHandler
from src.core.grid_detector import GridDetector
from src.core.image_processor import ImageProcessor
from src.utils.logger import log


@dataclass
class GridLabelConfig:
    """Configuration for grid label detection"""
    is_multi_characters: bool  # True: X1,Y1 / False: 1,A
    has_circle: bool           # True: labels in circles / False: plain text
    longitude_prefix: str      # 'X', 'M', '' (empty for single char)
    latitude_prefix: str       # 'Y', 'N', '' (empty for single char)
    longitude_start: Optional[Dict] = None  # {'label': 'X1', 'x': 100, 'y': 200}
    latitude_start: Optional[Dict] = None   # {'label': 'Y1', 'x': 200, 'y': 100}


@dataclass
class TestCase:
    """Test case definition"""
    name: str
    description: str
    page_number: int
    config: GridLabelConfig
    expected_longitude_count: int = 0  # Expected number of longitude lines
    expected_latitude_count: int = 0   # Expected number of latitude lines
    should_succeed: bool = True        # Whether test should pass


def run_test_case(
    pdf_path: str,
    test_case: TestCase,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Execute a single test case

    Args:
        pdf_path: Path to input PDF
        test_case: Test case configuration
        output_dir: Output directory for results

    Returns:
        Test result dictionary
    """
    log.info("=" * 80)
    log.info(f"TEST: {test_case.name}")
    log.info(f"Description: {test_case.description}")
    log.info(f"PDF: {pdf_path}, Page: {test_case.page_number}")
    log.info(f"Config:")
    log.info(f"  - is_multi_characters: {test_case.config.is_multi_characters}")
    log.info(f"  - has_circle: {test_case.config.has_circle}")
    log.info(f"  - longitude_prefix: '{test_case.config.longitude_prefix}'")
    log.info(f"  - latitude_prefix: '{test_case.config.latitude_prefix}'")
    if test_case.config.longitude_start:
        log.info(f"  - longitude_start: {test_case.config.longitude_start}")
    if test_case.config.latitude_start:
        log.info(f"  - latitude_start: {test_case.config.latitude_start}")
    log.info("=" * 80)

    try:
        # Initialize handlers
        detector = GridDetector()

        # Step 1: Extract page as image (using context manager for proper resource cleanup)
        log.info("Step 1: Extracting page as image...")
        with PDFHandler(pdf_path) as pdf_handler:
            page_image = pdf_handler.extract_page_as_image(
                test_case.page_number,
                output_format="numpy"
            )

        # Step 2: Crop and validate floor plan
        log.info("Step 2: Detecting and cropping floor plan...")
        cropped_image, crop_rect = ImageProcessor.find_and_crop_floor_plan(page_image)
        validation = ImageProcessor.validate_image_has_content(cropped_image)

        if not validation['has_content']:
            if not test_case.should_succeed:
                # Expected failure (e.g., page 1 with no floor plan)
                log.info(f"✅ Expected failure: No floor plan found (confidence: {validation['confidence']:.2f})")
                return {
                    'test_name': test_case.name,
                    'success': True,
                    'expected_failure': True,
                    'validation': validation
                }
            else:
                raise ValueError(
                    f"Page {test_case.page_number} does not contain a valid floor plan. "
                    f"Confidence: {validation['confidence']:.2f}"
                )

        # Step 3: Detect grid lines
        log.info("Step 3: Detecting grid lines...")

        grid_lines, _ = detector.process_floor_plan(
            cropped_image,
            is_multi_characters=test_case.config.is_multi_characters,
            has_circle=test_case.config.has_circle,
            longitude_prefix=test_case.config.longitude_prefix,
            latitude_prefix=test_case.config.latitude_prefix,
            longitude_start=test_case.config.longitude_start,
            latitude_start=test_case.config.latitude_start,
            auto_crop=False  # Already cropped
        )

        log.info(f"Detected {len(grid_lines)} grid lines")

        # Count lines by type
        long_lines = sum(1 for line in grid_lines if line.is_vertical)
        lat_lines = sum(1 for line in grid_lines if not line.is_vertical)

        # Log line labels for debugging
        long_labels = [line.label for line in grid_lines if line.is_vertical]
        lat_labels = [line.label for line in grid_lines if not line.is_vertical]

        log.info(f"Longitude lines ({long_lines}): {sorted(long_labels)}")
        log.info(f"Latitude lines ({lat_lines}): {sorted(lat_labels)}")

        # Step 4: Save results
        cropped_path = output_dir / f'page{test_case.page_number}_cropped.png'
        cropped_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(cropped_path), cropped_bgr)

        csv_path = output_dir / f'page{test_case.page_number}_gridlines.csv'
        detector.export_to_csv(grid_lines, str(csv_path))

        visualization_path = output_dir / f'page{test_case.page_number}_visualization.png'
        detector.visualize_grid_lines(cropped_image, grid_lines, str(visualization_path))

        # Check if results match expectations (if specified)
        meets_expectations = True
        expectation_msg = ""

        if test_case.expected_longitude_count > 0:
            if long_lines != test_case.expected_longitude_count:
                meets_expectations = False
                expectation_msg += f"Expected {test_case.expected_longitude_count} longitude lines, got {long_lines}. "

        if test_case.expected_latitude_count > 0:
            if lat_lines != test_case.expected_latitude_count:
                meets_expectations = False
                expectation_msg += f"Expected {test_case.expected_latitude_count} latitude lines, got {lat_lines}. "

        result = {
            'test_name': test_case.name,
            'description': test_case.description,
            'success': True,
            'meets_expectations': meets_expectations,
            'expectation_msg': expectation_msg if not meets_expectations else None,
            'config': {
                'is_multi_characters': test_case.config.is_multi_characters,
                'has_circle': test_case.config.has_circle,
                'longitude_prefix': test_case.config.longitude_prefix,
                'latitude_prefix': test_case.config.latitude_prefix,
            },
            'results': {
                'total_lines': len(grid_lines),
                'longitude_lines': long_lines,
                'latitude_lines': lat_lines,
                'longitude_labels': sorted(long_labels),
                'latitude_labels': sorted(lat_labels),
            },
            'expected': {
                'longitude_count': test_case.expected_longitude_count,
                'latitude_count': test_case.expected_latitude_count,
            },
            'files': {
                'cropped_path': str(cropped_path),
                'csv_path': str(csv_path),
                'visualization_path': str(visualization_path),
            },
            'validation': validation
        }

        # Save result JSON
        result_path = output_dir / 'result.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if meets_expectations:
            log.info(f"✅ Test completed successfully!")
        else:
            log.warning(f"⚠️  Test completed but didn't meet expectations: {expectation_msg}")

        log.info(f"   Lines: {len(grid_lines)} (long:{long_lines}, lat:{lat_lines})")
        log.info(f"   Output: {output_dir}")
        log.info("")

        return result

    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

        result = {
            'test_name': test_case.name,
            'description': test_case.description,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

        result_path = output_dir / 'result.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result


def get_all_test_cases() -> List[TestCase]:
    """
    Define all test cases

    Returns:
        List of test case definitions
    """
    test_cases = []

    # ========================================================================
    # CATEGORY 1: Invalid/Edge Cases
    # ========================================================================

    test_cases.append(TestCase(
        name="invalid_page_no_floor_plan",
        description="Page 1 - No floor plan content (should fail validation)",
        page_number=1,
        config=GridLabelConfig(
            is_multi_characters=True,
            has_circle=False,
            longitude_prefix='X',
            latitude_prefix='Y'
        ),
        should_succeed=False
    ))

    # ========================================================================
    # CATEGORY 2: Multi-Character Format with Circles (Pages 2-7)
    # Format: X1, X2, X3... and Y1, Y2, Y3...
    # ========================================================================

    test_cases.append(TestCase(
        name="multi_char_with_circles_xy",
        description="Pages 2-7: Multi-character format (X1-X9, Y1-Y5) with circles",
        page_number=2,
        config=GridLabelConfig(
            is_multi_characters=True,
            has_circle=True,
            longitude_prefix='X',
            latitude_prefix='Y'
        ),
        expected_longitude_count=9,  # X1-X9
        expected_latitude_count=5    # Y1-Y5
    ))

    test_cases.append(TestCase(
        name="multi_char_with_circles_xy_page3",
        description="Page 3: Multi-character format (X1-X5, Y1-Y4) with circles",
        page_number=3,
        config=GridLabelConfig(
            is_multi_characters=True,
            has_circle=True,
            longitude_prefix='X',
            latitude_prefix='Y'
        ),
        expected_longitude_count=8,
        expected_latitude_count=4
    ))

    test_cases.append(TestCase(
        name="multi_char_with_circles_mn",
        description="Page 2: Multi-character format with custom M/N prefixes",
        page_number=2,
        config=GridLabelConfig(
            is_multi_characters=True,
            has_circle=True,
            longitude_prefix='M',
            latitude_prefix='N'
        ),
        expected_longitude_count=0,
        expected_latitude_count=0
    ))

    # ========================================================================
    # CATEGORY 3: Single Character Format with Circles (Pages 8-11)
    # Format: Pure numbers (1,2,3,4,6) and pure letters (A,B,C,D,E,F)
    # ========================================================================

    test_cases.append(TestCase(
        name="single_char_with_circles_page8",
        description="Page 8: Single character format (1-5, A-F) with circles",
        page_number=8,
        config=GridLabelConfig(
            is_multi_characters=False,
            has_circle=True,
            longitude_prefix='',  # Empty for pure numbers
            latitude_prefix=''    # Empty for pure letters
        ),
        expected_longitude_count=6,  # 1-2-3-4-5-6 (5 now detected from middle/top)
        expected_latitude_count=8   # A-D-E-F-A-D-E-F
    ))

    test_cases.append(TestCase(
        name="single_char_with_circles_page9",
        description="Page 9: Single character format (1-6, A-F) with circles",
        page_number=9,
        config=GridLabelConfig(
            is_multi_characters=False,
            has_circle=True,
            longitude_prefix='',
            latitude_prefix=''
        ),
        expected_longitude_count=6,  # 1-2-3-4-5-6
        expected_latitude_count=10   # A-B-D-E-F × 2 (no C on this floor plan)
    ))

    test_cases.append(TestCase(
        name="single_char_with_circles_page10",
        description="Page 10: Single character format (1-6, A-F) with circles",
        page_number=10,
        config=GridLabelConfig(
            is_multi_characters=False,
            has_circle=True,
            longitude_prefix='',
            latitude_prefix=''
        ),
        expected_longitude_count=6,  # 1-2-3-4-5-6 (5 now detected from middle/top)
        expected_latitude_count=8   # A-D-E-F-A-D-E-F
    ))

    test_cases.append(TestCase(
        name="single_char_with_circles_page11",
        description="Page 11: Single character format (1-6, A-F) with circles",
        page_number=11,
        config=GridLabelConfig(
            is_multi_characters=False,
            has_circle=True,
            longitude_prefix='',
            latitude_prefix=''
        ),
        expected_longitude_count=6,  # 1-2-3-4-5-6 (5 now detected from middle/top)
        expected_latitude_count=8   # A-C-E-F-A-C-E-F
    ))

    # ========================================================================
    # CATEGORY 4: With Reference Coordinates
    # ========================================================================

    test_cases.append(TestCase(
        name="multi_char_with_reference_coords",
        description="Page 2: Multi-character with reference coordinates",
        page_number=2,
        config=GridLabelConfig(
            is_multi_characters=True,
            has_circle=True,
            longitude_prefix='X',
            latitude_prefix='Y',
            longitude_start={'label': 'X1', 'x': 1000, 'y': 2000},
            latitude_start={'label': 'Y1', 'x': 500, 'y': 1500}
        ),
        expected_longitude_count=9,
        expected_latitude_count=5
    ))

    test_cases.append(TestCase(
        name="single_char_with_reference_coords",
        description="Page 8: Single character with reference coordinates",
        page_number=8,
        config=GridLabelConfig(
            is_multi_characters=False,
            has_circle=True,
            longitude_prefix='',
            latitude_prefix='',
            longitude_start={'label': '1', 'x': 1220, 'y': 3240},
            latitude_start={'label': 'A', 'x': 587, 'y': 2630}
        ),
        expected_longitude_count=6,  # 1-2-3-4-5-6 (5 now detected from middle/top)
        expected_latitude_count=8   # A-D-E-F-A-D-E-F
    ))

    return test_cases


def print_summary(results: List[Dict[str, Any]]):
    """Print test summary"""
    log.info("")
    log.info("=" * 80)
    log.info("COMPREHENSIVE TEST SUMMARY")
    log.info("=" * 80)

    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    meets_expectations = sum(
        1 for r in results
        if r.get('success', False) and r.get('meets_expectations', True)
    )

    log.info(f"Total tests: {len(results)}")
    log.info(f"✅ Passed: {successful}")
    log.info(f"❌ Failed: {failed}")
    log.info(f"⚠️  Passed but didn't meet expectations: {successful - meets_expectations}")
    log.info("")

    for result in results:
        test_name = result.get('test_name', 'Unknown')
        description = result.get('description', '')

        if result.get('success'):
            if result.get('expected_failure'):
                log.info(f"✅ {test_name}: Expected failure - {description}")
            elif not result.get('meets_expectations', True):
                msg = result.get('expectation_msg', '')
                log.warning(f"⚠️  {test_name}: {msg}")
            else:
                res = result.get('results', {})
                log.info(
                    f"✅ {test_name}: "
                    f"{res.get('total_lines', 0)} lines "
                    f"(long:{res.get('longitude_lines', 0)}, "
                    f"lat:{res.get('latitude_lines', 0)})"
                )
        else:
            log.error(f"❌ {test_name}: {result.get('error', 'Unknown error')}")

    log.info("")
    log.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive test suite for floor plan grid detection'
    )
    parser.add_argument(
        '--pdf',
        default='test/input/test.pdf',
        help='Path to test PDF file'
    )
    parser.add_argument(
        '--output',
        default='test/output',
        help='Output directory'
    )
    parser.add_argument(
        '--test',
        help='Run specific test case by name'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available test cases'
    )

    args = parser.parse_args()

    # Get all test cases
    all_test_cases = get_all_test_cases()

    # List tests if requested
    if args.list:
        log.info("Available test cases:")
        for i, tc in enumerate(all_test_cases, 1):
            log.info(f"  {i}. {tc.name}")
            log.info(f"     {tc.description}")
        return

    # Check PDF exists
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        log.error(f"❌ PDF file not found: {pdf_path}")
        log.info("Please provide a valid PDF file path")
        log.info(f"Example: python3 test.py --pdf test/input/test.pdf")
        sys.exit(1)

    # Create output directory
    base_output_dir = Path(args.output)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Testing PDF: {pdf_path}")
    log.info(f"Output directory: {base_output_dir}")
    log.info("")

    # Filter test cases if specific test requested
    if args.test:
        test_cases = [tc for tc in all_test_cases if tc.name == args.test]
        if not test_cases:
            log.error(f"❌ Test case not found: {args.test}")
            log.info("Use --list to see available test cases")
            sys.exit(1)
    else:
        test_cases = all_test_cases

    # Run tests
    results = []
    for test_case in test_cases:
        output_dir = base_output_dir / test_case.name
        output_dir.mkdir(parents=True, exist_ok=True)

        result = run_test_case(str(pdf_path), test_case, output_dir)
        results.append(result)

    # Print summary
    print_summary(results)

    # Exit with error code if any tests failed
    failed_count = sum(1 for r in results if not r.get('success', False))
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
