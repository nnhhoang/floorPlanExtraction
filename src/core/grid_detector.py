"""Grid line detection and coordinate extraction module"""
import os
import cv2
import numpy as np
import pandas as pd
import easyocr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.utils.logger import log
from src.core.image_processor import ImageProcessor


@dataclass
class GridLine:
    """Represents a detected grid line."""
    label: str  # e.g., "X1", "Y1"
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    angle: float  # Line angle in degrees
    is_vertical: bool  # True for X-axis (vertical), False for Y-axis (horizontal)


class GridDetector:
    """Detect grid lines and extract coordinates from floor plan images."""

    def __init__(self, use_gpu: bool = True):
        """
        Initialize grid detector.

        Args:
            use_gpu: Use GPU for OCR if available (default: True)
        """
        # Initialize EasyOCR for grid label detection (X1, Y1, etc.)
        # Use English for Latin character (X, Y) recognition
        # EasyOCR will automatically fallback to CPU if GPU not available
        try:
            self.ocr = easyocr.Reader(['en'], gpu=use_gpu)
            gpu_status = 'GPU' if use_gpu else 'CPU'
            log.info(f"Grid detector initialized with EasyOCR ({gpu_status})")
        except Exception as e:
            log.error(f"Failed to initialize EasyOCR: {e}")
            raise
        log.info("Grid detector ready for X/Y label detection")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for grid line detection.

        Args:
            image: Input image (BGR or RGB)

        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        log.debug("Image preprocessed for grid detection")
        return binary

    def detect_lines(
        self,
        binary_image: np.ndarray,
        min_line_length: int = 100,
        max_line_gap: int = 10
    ) -> List[np.ndarray]:
        """
        Detect lines using Probabilistic Hough Transform.

        Args:
            binary_image: Binary image
            min_line_length: Minimum line length in pixels
            max_line_gap: Maximum gap between line segments

        Returns:
            List of detected lines [x1, y1, x2, y2]
        """
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            binary_image,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        if lines is None:
            log.warning("No lines detected in image")
            return []

        lines_list = [line[0] for line in lines]
        log.info(f"Detected {len(lines_list)} lines using Hough Transform")
        return lines_list

    def filter_main_grid_lines(
        self,
        lines: List[np.ndarray],
        image_shape: Tuple[int, int],
        min_length_ratio: float = 0.6,
        exclude_border: bool = True,
        border_margin: int = 50
    ) -> List[np.ndarray]:
        """
        Filter lines to keep only main grid lines (long lines spanning most of the image).
        Optionally excludes border/frame lines.

        Args:
            lines: List of detected lines
            image_shape: Shape of the image (height, width)
            min_length_ratio: Minimum length ratio relative to image dimension (0.6 = 60%)
            exclude_border: If True, exclude lines near image borders (outer frame)
            border_margin: Distance from border to consider a line as border line

        Returns:
            Filtered list of main grid lines
        """
        if not lines:
            return []

        height, width = image_shape
        filtered_lines = []

        for line in lines:
            x1, y1, x2, y2 = line
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Check if this is a border line (too close to edges)
            if exclude_border:
                is_border = False

                # For vertical lines, use smaller margin (more lenient)
                if 80 <= angle <= 100:  # Vertical
                    vertical_margin = border_margin // 2  # Half the margin for vertical lines
                    if x1 < vertical_margin or x1 > width - vertical_margin:
                        is_border = True
                # For horizontal lines, check if near top or bottom edge
                elif angle <= 10 or angle >= 170:  # Horizontal
                    if y1 < border_margin or y1 > height - border_margin:
                        is_border = True

                if is_border:
                    log.debug(f"Excluding border line at ({x1}, {y1})")
                    continue

            # For vertical lines, check length against height (use more lenient threshold)
            if 80 <= angle <= 100:  # Vertical
                # Vertical lines may not span full height, so use 90% of the threshold
                vertical_threshold = min_length_ratio * 0.9
                if length >= height * vertical_threshold:
                    filtered_lines.append(line)
            # For horizontal lines, check length against width
            elif angle <= 10 or angle >= 170:  # Horizontal
                if length >= width * min_length_ratio:
                    filtered_lines.append(line)

        log.info(f"Filtered to {len(filtered_lines)} main grid lines (from {len(lines)})")
        return filtered_lines

    def classify_lines(
        self,
        lines: List[np.ndarray],
        angle_threshold: float = 10.0
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Classify lines as vertical (X-axis) or horizontal (Y-axis).

        Args:
            lines: List of lines [x1, y1, x2, y2]
            angle_threshold: Angle threshold in degrees to classify as vertical/horizontal

        Returns:
            Tuple of (vertical_lines, horizontal_lines)
        """
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line

            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Classify based on angle
            # Vertical lines: angle close to 90 degrees
            # Horizontal lines: angle close to 0 or 180 degrees
            if 90 - angle_threshold <= angle <= 90 + angle_threshold:
                vertical_lines.append(line)
            elif angle <= angle_threshold or angle >= 180 - angle_threshold:
                horizontal_lines.append(line)

        log.info(
            f"Classified lines: {len(vertical_lines)} vertical, "
            f"{len(horizontal_lines)} horizontal"
        )
        return vertical_lines, horizontal_lines

    def merge_collinear_lines(
        self,
        lines: List[np.ndarray],
        is_vertical: bool,
        distance_threshold: int = 10
    ) -> List[np.ndarray]:
        """
        Merge collinear lines that are close to each other.

        Args:
            lines: List of lines to merge
            is_vertical: True if lines are vertical, False if horizontal
            distance_threshold: Maximum distance between lines to merge

        Returns:
            List of merged lines
        """
        if not lines:
            return []

        merged_lines = []
        lines = sorted(lines, key=lambda l: l[0] if is_vertical else l[1])

        current_group = [lines[0]]

        for line in lines[1:]:
            x1, y1, x2, y2 = line

            # Get reference position (x for vertical, y for horizontal)
            if is_vertical:
                ref_pos = x1
                prev_ref = current_group[0][0]
            else:
                ref_pos = y1
                prev_ref = current_group[0][1]

            # Check if line is close to current group
            if abs(ref_pos - prev_ref) <= distance_threshold:
                current_group.append(line)
            else:
                # Merge current group and start new one
                merged_lines.append(self._merge_line_group(current_group, is_vertical))
                current_group = [line]

        # Merge last group
        if current_group:
            merged_lines.append(self._merge_line_group(current_group, is_vertical))

        log.debug(f"Merged {len(lines)} lines into {len(merged_lines)} lines")
        return merged_lines

    def _merge_line_group(
        self,
        line_group: List[np.ndarray],
        is_vertical: bool
    ) -> np.ndarray:
        """
        Merge a group of collinear lines into one line.

        Args:
            line_group: Group of lines to merge
            is_vertical: True if lines are vertical

        Returns:
            Merged line [x1, y1, x2, y2]
        """
        if len(line_group) == 1:
            return line_group[0]

        # Calculate average position and extent
        if is_vertical:
            x = int(np.mean([line[0] for line in line_group]))
            y_min = min(min(line[1], line[3]) for line in line_group)
            y_max = max(max(line[1], line[3]) for line in line_group)
            return np.array([x, y_min, x, y_max])
        else:
            y = int(np.mean([line[1] for line in line_group]))
            x_min = min(min(line[0], line[2]) for line in line_group)
            x_max = max(max(line[0], line[2]) for line in line_group)
            return np.array([x_min, y, x_max, y])

    def detect_grid_labels(
        self,
        image: np.ndarray,
        lines: List[np.ndarray],
        is_vertical: bool,
        search_margin: int = 100
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Detect labels (X1, X2, Y1, Y2, etc.) near grid lines using OCR.

        Args:
            image: Original image
            lines: List of lines to label
            is_vertical: True if lines are vertical (X-axis)
            search_margin: Pixel margin to search for labels (increased for circled labels)

        Returns:
            List of (label, line) tuples
        """
        labeled_lines = []
        height, width = image.shape[:2]

        for line in lines:
            x1, y1, x2, y2 = line

            # Define search region based on line orientation
            # Search near the top/bottom for vertical lines, left/right for horizontal lines
            if is_vertical:
                # Search near top and bottom of the line (where labels typically are)
                roi_x1 = max(0, x1 - search_margin)
                roi_x2 = min(width, x1 + search_margin)
                # Search at top portion
                roi_y1_top = max(0, min(y1, y2))
                roi_y2_top = min(height, min(y1, y2) + search_margin * 2)
                roi_top = image[roi_y1_top:roi_y2_top, roi_x1:roi_x2]

                # Try OCR on top region
                label = self._ocr_label_from_roi(roi_top, is_vertical)

                if not label:
                    # Try bottom region
                    roi_y1_bot = max(0, max(y1, y2) - search_margin * 2)
                    roi_y2_bot = min(height, max(y1, y2))
                    roi_bot = image[roi_y1_bot:roi_y2_bot, roi_x1:roi_x2]
                    label = self._ocr_label_from_roi(roi_bot, is_vertical)
            else:
                # Search left and right of the line
                roi_y1 = max(0, y1 - search_margin)
                roi_y2 = min(height, y1 + search_margin)
                # Search at left portion
                roi_x1_left = max(0, min(x1, x2))
                roi_x2_left = min(width, min(x1, x2) + search_margin * 2)
                roi_left = image[roi_y1:roi_y2, roi_x1_left:roi_x2_left]

                label = self._ocr_label_from_roi(roi_left, is_vertical)

                if not label:
                    # Try right region
                    roi_x1_right = max(0, max(x1, x2) - search_margin * 2)
                    roi_x2_right = min(width, max(x1, x2))
                    roi_right = image[roi_y1:roi_y2, roi_x1_right:roi_x2_right]
                    label = self._ocr_label_from_roi(roi_right, is_vertical)

            # If label found, use it; otherwise generate default
            if label:
                labeled_lines.append((label, line))
            else:
                prefix = "X" if is_vertical else "Y"
                label = f"{prefix}{len(labeled_lines) + 1}"
                labeled_lines.append((label, line))

        log.info(f"Labeled {len(labeled_lines)} grid lines")
        return labeled_lines

    def _ocr_label_from_roi(self, roi: np.ndarray, is_vertical: bool) -> Optional[str]:
        """
        Perform OCR on ROI to extract grid label.

        Args:
            roi: Region of interest image
            is_vertical: True if expecting X labels, False for Y labels

        Returns:
            Extracted label or None
        """
        if roi.size == 0:
            return None

        try:
            # EasyOCR returns: [(bbox, text, confidence), ...]
            result = self.ocr.readtext(roi)

            if result:
                # Extract text from EasyOCR result
                texts = [detection[1] for detection in result]
                full_text = ''.join(texts)

                # Look for X or Y followed by numbers
                label = self._extract_grid_label(full_text, is_vertical)
                return label

        except Exception as e:
            log.debug(f"OCR failed for ROI: {e}")

        return None

    def _extract_grid_label(self, text: str, is_vertical: bool) -> Optional[str]:
        """
        Extract grid label from OCR text.

        Args:
            text: OCR text
            is_vertical: True if expecting X labels, False for Y labels

        Returns:
            Extracted label or None
        """
        import re

        # Expected prefix
        prefix = "X" if is_vertical else "Y"

        # Look for pattern: X/Y followed by digits
        pattern = rf'{prefix}(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return f"{prefix}{match.group(1)}"

        return None

    def detect_label_circles(
        self,
        image: np.ndarray,
        min_radius: int = 15,
        max_radius: int = 50
    ) -> List[Tuple[int, int, int]]:
        """
        Detect circles with black borders that contain grid labels.
        Uses contour-based detection for better accuracy.

        Args:
            image: Input image
            min_radius: Minimum circle radius in pixels
            max_radius: Maximum circle radius in pixels

        Returns:
            List of (x, y, radius) tuples for detected circles
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Try multiple threshold values to catch circles with different intensities
        detected_circles = []
        seen_positions = set()

        for thresh_val in [200, 180, 160, 220]:
            # Apply binary thresholding to get black shapes
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Calculate area and perimeter
                area = cv2.contourArea(contour)
                if area < np.pi * min_radius**2 or area > np.pi * max_radius**2:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                # Calculate circularity: 4*pi*area / perimeter^2
                # Perfect circle = 1.0, square = 0.785
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # More lenient circularity threshold
                if circularity > 0.65:
                    # Get minimum enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    x, y, radius = int(x), int(y), int(radius)

                    # Verify size constraints
                    if min_radius <= radius <= max_radius:
                        # Check if we've already found a circle at this position
                        pos_key = (x // 10, y // 10)  # Grid to avoid duplicates
                        if pos_key not in seen_positions:
                            detected_circles.append((x, y, radius))
                            seen_positions.add(pos_key)
                            log.debug(f"Found circle at ({x}, {y}) with radius {radius}, circularity {circularity:.2f}")

        log.info(f"Detected {len(detected_circles)} potential label circles using contour method")
        return detected_circles

    def _verify_circle_border(
        self,
        edges: np.ndarray,
        x: int,
        y: int,
        radius: int,
        threshold: float = 0.3
    ) -> bool:
        """
        Verify that a circle has a strong border by checking edge pixels.

        Args:
            edges: Edge-detected image
            x, y: Circle center
            radius: Circle radius
            threshold: Minimum ratio of edge pixels on circle perimeter

        Returns:
            True if circle has strong border
        """
        # Sample points on circle perimeter
        num_samples = 32
        edge_count = 0

        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            px = int(x + radius * np.cos(angle))
            py = int(y + radius * np.sin(angle))

            # Check if point is within image bounds
            if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
                # Check if there's an edge at this point (or nearby)
                if edges[py, px] > 0:
                    edge_count += 1

        # Return True if enough edge pixels found
        ratio = edge_count / num_samples
        return ratio >= threshold

    def extract_label_from_circle(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        radius: int,
        padding: int = 10,
        longitude_prefix: str = 'X',
        latitude_prefix: str = 'Y'
    ) -> Optional[str]:
        """
        Extract text label from a circle region using OCR.
        Tries multiple preprocessing approaches to maximize OCR success.

        Supports multiple label formats:
        - Prefixed: X1, Y1, M1, N1 (longitude_prefix='X', latitude_prefix='Y')
        - Pure numbers: 1, 2, 3... (longitude_prefix='', numbers for longitude)
        - Pure letters: A, B, C... (latitude_prefix='', letters for latitude)

        Args:
            image: Input image
            x, y: Circle center coordinates
            radius: Circle radius
            padding: Extra padding around circle
            longitude_prefix: Prefix for longitude axis (default 'X', use '' for pure numbers)
            latitude_prefix: Prefix for latitude axis (default 'Y', use '' for pure letters)

        Returns:
            Extracted label (e.g., "X1", "Y2", "M1", "N2", "1", "A") or None
        """
        height, width = image.shape[:2]

        # Define ROI around circle with padding
        roi_x1 = max(0, x - radius - padding)
        roi_x2 = min(width, x + radius + padding)
        roi_y1 = max(0, y - radius - padding)
        roi_y2 = min(height, y + radius + padding)

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            return None

        import re

        # Determine pattern based on prefix settings
        if longitude_prefix == '' and latitude_prefix == '':
            # Pure number/letter format: numbers for longitude, letters for latitude
            # Match either: pure digits (1, 2, 3...) OR pure letters (A, B, C...)
            pattern = rf'([A-Z])|(\d{{1,2}})'  # Limit to 1-2 digits
        elif longitude_prefix == '':
            # Pure numbers for longitude, prefixed for latitude
            lat_pattern = latitude_prefix.upper() + latitude_prefix.lower()
            pattern = rf'([{lat_pattern}])(\d{{1,2}})|(\d{{1,2}})'  # Limit to 1-2 digits
        elif latitude_prefix == '':
            # Prefixed for longitude, pure letters for latitude
            long_pattern = longitude_prefix.upper() + longitude_prefix.lower()
            pattern = rf'([{long_pattern}])(\d{{1,2}})|([A-Z])'  # Limit to 1-2 digits
        else:
            # Standard prefixed format (X1, Y1 or M1, N1, etc.)
            # For custom prefixes (M/N), limit to 1-2 digits to prevent OCR concatenation
            long_pattern = longitude_prefix.upper() + longitude_prefix.lower()
            lat_pattern = latitude_prefix.upper() + latitude_prefix.lower()
            pattern = rf'([{long_pattern}{lat_pattern}])(\d{{1,2}})'  # Limit to 1-2 digits

        # Keep as color image (EasyOCR handles 3 channels BGR/RGB)
        # Use single scale factor for speed - 4x is good balance of accuracy and speed
        scale_factors = [4]

        # Use only normal preprocessing for speed
        preprocessing_strategies = [
            ('normal', lambda img: img),
        ]

        for scale_factor in scale_factors:
            # Resize ROI for better OCR (larger text is easier to recognize)
            h, w = roi.shape[:2]
            enlarged = cv2.resize(
                roi,
                (w * scale_factor, h * scale_factor),
                interpolation=cv2.INTER_CUBIC
            )

            for strategy_name, preprocess_fn in preprocessing_strategies:
                try:
                    # Apply preprocessing
                    preprocessed = preprocess_fn(enlarged)

                    # Run EasyOCR - returns: [(bbox, text, confidence), ...]
                    result = self.ocr.readtext(preprocessed)

                    if not result:
                        log.debug(f"No OCR result for circle at ({x}, {y}) scale={scale_factor}x {strategy_name}")
                        continue

                    # Join all detected texts from all detections
                    full_text = ''
                    texts_found = []
                    for detection in result:
                        # EasyOCR format: (bbox, text, confidence)
                        if len(detection) >= 3:
                            bbox, text, confidence = detection[0], detection[1], detection[2]
                            full_text += str(text).strip()
                            texts_found.append(f"{text}({confidence:.2f})")

                    # Debug: Log what OCR found
                    if texts_found:
                        log.debug(f"OCR at ({x}, {y}) scale={scale_factor}x {strategy_name} found: {', '.join(texts_found)} -> full_text='{full_text}'")

                    # Try to extract label based on format
                    match = re.search(pattern, full_text, re.IGNORECASE)

                    if match:
                        # Handle different pattern match groups based on prefix settings
                        if longitude_prefix == '' and latitude_prefix == '':
                            # Pure number/letter format
                            if match.group(1):  # Letter matched (latitude)
                                label = match.group(1).upper()
                            elif match.group(2):  # Number matched (longitude)
                                label = match.group(2)
                            else:
                                continue
                        elif longitude_prefix == '':
                            # Pure numbers for longitude OR prefixed latitude
                            if match.group(3):  # Pure number matched
                                label = match.group(3)
                            elif match.group(1) and match.group(2):  # Prefixed latitude
                                prefix = match.group(1).upper()
                                number = match.group(2)
                                label = f"{prefix}{number}"
                            else:
                                continue
                        elif latitude_prefix == '':
                            # Prefixed longitude OR pure letter latitude
                            if match.group(3):  # Pure letter matched
                                label = match.group(3).upper()
                            elif match.group(1) and match.group(2):  # Prefixed longitude
                                prefix = match.group(1).upper()
                                number = match.group(2)
                                label = f"{prefix}{number}"
                            else:
                                continue
                        else:
                            # Standard prefixed format
                            prefix = match.group(1).upper()
                            number = match.group(2)
                            label = f"{prefix}{number}"

                        # Apply OCR validation to fix common misreads (T→1, I→1, O→0)
                        is_multi_characters = (longitude_prefix != '' or latitude_prefix != '')
                        validated_label = self.validate_ocr_label(
                            label,
                            is_multi_characters=is_multi_characters,
                            longitude_prefix=longitude_prefix,
                            latitude_prefix=latitude_prefix
                        )

                        log.info(f"Extracted label '{validated_label}' from circle at ({x}, {y}) scale={scale_factor}x {strategy_name}")
                        return validated_label
                    else:
                        # Pattern didn't match - try fallback validation for common misreads
                        if full_text:
                            log.debug(f"No label pattern match in text '{full_text}' at ({x}, {y})")

                            # For single-character format, try validating the full text as-is
                            if longitude_prefix == '' and latitude_prefix == '':
                                # Check if it's a common misread that we can fix
                                validated = self.validate_ocr_label(
                                    full_text,
                                    is_multi_characters=False,
                                    longitude_prefix=longitude_prefix,
                                    latitude_prefix=latitude_prefix
                                )
                                # If validation changed it, and it's now a valid single char, use it
                                if validated != full_text and (validated.isdigit() or validated.isalpha()):
                                    log.info(f"Fallback validation: '{full_text}' → '{validated}' at ({x}, {y}) {strategy_name}")
                                    return validated

                except Exception as e:
                    log.debug(f"OCR scale={scale_factor}x {strategy_name} failed for circle at ({x}, {y}): {e}")
                    continue

        return None

    def filter_labeled_positions_by_alignment(
        self,
        labeled_positions: List[Tuple[str, int, int]],
        alignment_tolerance: int = 20,
        longitude_prefix: str = 'X',
        latitude_prefix: str = 'Y'
    ) -> List[Tuple[str, int, int]]:
        """
        Filter labeled positions based on alignment rules:
        - Longitude labels (X1, X2... or M1, M2...) should have the same Y coordinate (horizontal alignment)
        - Latitude labels (Y1, Y2... or N1, N2...) should have the same X coordinate (vertical alignment)

        This significantly reduces false positives from circle detection.

        Args:
            labeled_positions: List of (label, x, y) tuples
            alignment_tolerance: Pixel tolerance for alignment (default: 20px)
            longitude_prefix: Prefix for longitude axis (default 'X')
            latitude_prefix: Prefix for latitude axis (default 'Y')

        Returns:
            Filtered list of (label, x, y) tuples that follow alignment rules
        """
        if not labeled_positions:
            return []

        from collections import defaultdict

        # Separate longitude and latitude labels based on prefix settings
        if longitude_prefix == '' and latitude_prefix == '':
            # Pure number/letter format: numbers are longitude, letters are latitude
            long_labels = [(label, x, y) for label, x, y in labeled_positions if label.isdigit()]
            lat_labels = [(label, x, y) for label, x, y in labeled_positions if label.isalpha()]
        elif longitude_prefix == '':
            # Pure numbers for longitude, prefixed for latitude
            long_labels = [(label, x, y) for label, x, y in labeled_positions if label.isdigit()]
            lat_labels = [(label, x, y) for label, x, y in labeled_positions if label.startswith(latitude_prefix.upper())]
        elif latitude_prefix == '':
            # Prefixed for longitude, pure letters for latitude
            long_labels = [(label, x, y) for label, x, y in labeled_positions if label.startswith(longitude_prefix.upper())]
            lat_labels = [(label, x, y) for label, x, y in labeled_positions if label.isalpha()]
        else:
            # Standard prefixed format
            long_labels = [(label, x, y) for label, x, y in labeled_positions if label.startswith(longitude_prefix.upper())]
            lat_labels = [(label, x, y) for label, x, y in labeled_positions if label.startswith(latitude_prefix.upper())]

        filtered_positions = []

        # Process longitude labels (should be horizontally aligned - same Y coordinate)
        if long_labels:
            if longitude_prefix == '':
                # Single-char format: could be numbers OR letters depending on floor plan convention
                long_label_type = f"vertical-axis ({len(long_labels)} labels)"
            else:
                long_label_type = longitude_prefix
            log.info(f"Filtering {len(long_labels)} {long_label_type} labels by horizontal alignment...")

            # Group longitude labels by Y coordinate (with tolerance)
            y_groups = defaultdict(list)
            for label, x, y in long_labels:
                # Find existing group within tolerance
                found_group = False
                for group_y in y_groups.keys():
                    if abs(y - group_y) <= alignment_tolerance:
                        y_groups[group_y].append((label, x, y))
                        found_group = True
                        break
                if not found_group:
                    y_groups[y].append((label, x, y))

            # Find the largest group (most aligned longitude labels)
            if y_groups:
                largest_group_y = max(y_groups.keys(), key=lambda k: len(y_groups[k]))
                largest_group = y_groups[largest_group_y]

                if longitude_prefix == '':
                    long_label_type = "vertical-axis"
                else:
                    long_label_type = longitude_prefix
                log.info(
                    f"Found {len(largest_group)} horizontally aligned {long_label_type} labels "
                    f"at Y≈{largest_group_y} (from {len(y_groups)} groups)"
                )
                filtered_positions.extend(largest_group)
            else:
                if longitude_prefix == '':
                    long_label_type = "vertical-axis"
                else:
                    long_label_type = longitude_prefix
                log.warning(f"No aligned {long_label_type} label groups found")

        # Process latitude labels (should be vertically aligned - same X coordinate)
        if lat_labels:
            if latitude_prefix == '':
                # Single-char format: could be letters OR numbers depending on floor plan convention
                lat_label_type = f"horizontal-axis ({len(lat_labels)} labels)"
            else:
                lat_label_type = latitude_prefix
            log.info(f"Filtering {len(lat_labels)} {lat_label_type} labels by vertical alignment...")

            # Group latitude labels by X coordinate (with tolerance)
            x_groups = defaultdict(list)
            for label, x, y in lat_labels:
                # Find existing group within tolerance
                found_group = False
                for group_x in x_groups.keys():
                    if abs(x - group_x) <= alignment_tolerance:
                        x_groups[group_x].append((label, x, y))
                        found_group = True
                        break
                if not found_group:
                    x_groups[x].append((label, x, y))

            # Find the largest group (most aligned latitude labels)
            if x_groups:
                largest_group_x = max(x_groups.keys(), key=lambda k: len(x_groups[k]))
                largest_group = x_groups[largest_group_x]

                if latitude_prefix == '':
                    lat_label_type = "horizontal-axis"
                else:
                    lat_label_type = latitude_prefix
                log.info(
                    f"Found {len(largest_group)} vertically aligned {lat_label_type} labels "
                    f"at X≈{largest_group_x} (from {len(x_groups)} groups)"
                )
                filtered_positions.extend(largest_group)
            else:
                if latitude_prefix == '':
                    lat_label_type = "horizontal-axis"
                else:
                    lat_label_type = latitude_prefix
                log.warning(f"No aligned {lat_label_type} label groups found")

        log.info(
            f"Alignment filtering: {len(labeled_positions)} → {len(filtered_positions)} labels "
            f"({len(labeled_positions) - len(filtered_positions)} removed)"
        )

        return filtered_positions

    def _apply_binary_threshold(
        self,
        gray_image: np.ndarray,
        method: str = "otsu"
    ) -> np.ndarray:
        """
        Apply binary thresholding to grayscale image.

        Args:
            gray_image: Grayscale image
            method: "otsu" or "adaptive"

        Returns:
            Binary image
        """
        if method == "otsu":
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            return binary
        else:
            return gray_image

    def detect_grid_labels_by_margin_ocr(
        self,
        image: np.ndarray,
        x_margin_height: int = 200,
        y_margin_width: int = 200,
        enlarge_factor: int = 2,  # Reduced from 3 for speed
        longitude_prefix: str = 'X',
        latitude_prefix: str = 'Y'
    ) -> List[Tuple[str, int, int]]:
        """
        Detect grid labels by scanning MARGINS with OCR + enlargement.
        No circle dependency - scans edges where labels typically appear.

        Strategy:
        - Longitude labels (X1, X2... or M1, M2...) are typically at BOTTOM margin (horizontally aligned)
        - Latitude labels (Y1, Y2... or N1, N2...) are typically at LEFT margin (vertically aligned)
        - Enlarge margin regions before OCR for better accuracy on small text

        Args:
            image: Input image
            x_margin_height: Height of bottom margin to scan for longitude labels (default 200px)
            y_margin_width: Width of left margin to scan for latitude labels (default 200px)
            enlarge_factor: Factor to enlarge margin regions (default 3x)
            longitude_prefix: Prefix for longitude axis (default 'X')
            latitude_prefix: Prefix for latitude axis (default 'Y')

        Returns:
            List of (label, x, y) tuples where x, y is the center position in original image
        """
        try:
            height, width = image.shape[:2]
            labeled_positions = []
            import re

            log.info(
                f"Scanning margins for grid labels (longitude={longitude_prefix}, latitude={latitude_prefix}, "
                f"X margin: {x_margin_height}px, Y margin: {y_margin_width}px, enlarge={enlarge_factor}x)..."
            )

            # Define margins to scan with specific dimensions for longitude and latitude labels
            # ONLY scan bottom and left to avoid false positives from interior content
            margins = {
                'bottom': (0, height - x_margin_height, width, height),  # Longitude labels: bottom x_margin_height
                'left': (0, 0, y_margin_width, height),                   # Latitude labels: left y_margin_width
            }

            for margin_name, (x1, y1, x2, y2) in margins.items():
                # Extract margin region
                margin_roi = image[y1:y2, x1:x2]

                if margin_roi.size == 0:
                    continue

                # Enlarge margin for better OCR
                h, w = margin_roi.shape[:2]
                enlarged = cv2.resize(
                    margin_roi,
                    (w * enlarge_factor, h * enlarge_factor),
                    interpolation=cv2.INTER_CUBIC
                )

                log.info(f"Scanning {margin_name} margin ({w}x{h} → {w*enlarge_factor}x{h*enlarge_factor})...")

                # Run EasyOCR on enlarged margin
                result = self.ocr.readtext(enlarged)

                if not result:
                    log.debug(f"No OCR results in {margin_name} margin")
                    continue

                log.debug(f"Found {len(result)} text detections in {margin_name} margin")

                # Process detections - EasyOCR format: [(bbox, text, confidence), ...]
                for detection in result:
                    try:
                        if len(detection) < 3:
                            continue

                        bbox, text, confidence = detection[0], detection[1], detection[2]
                        text_clean = str(text).strip().upper()

                        # Create pattern based on prefix settings
                        if longitude_prefix == '' and latitude_prefix == '':
                            # Pure number/letter format
                            pattern = rf'([A-Z])|(\d{{1,2}})'  # Limit to 1-2 digits
                        elif longitude_prefix == '':
                            # Pure numbers for longitude, prefixed for latitude
                            lat_pattern = latitude_prefix.upper() + latitude_prefix.lower()
                            pattern = rf'([{lat_pattern}])[^\w]*(\d{{1,2}})|(\d{{1,2}})'  # Limit to 1-2 digits
                        elif latitude_prefix == '':
                            # Prefixed for longitude, pure letters for latitude
                            long_pattern = longitude_prefix.upper() + longitude_prefix.lower()
                            pattern = rf'([{long_pattern}])[^\w]*(\d{{1,2}})|([A-Z])'  # Limit to 1-2 digits
                        else:
                            # Standard prefixed format (X1, Y1 or M1, N1, etc.)
                            # For custom prefixes (M/N), limit to 1-2 digits to prevent OCR concatenation
                            pattern = rf'([{longitude_prefix.upper()}{latitude_prefix.upper()}])[^\w]*(\d{{1,2}})'  # Limit to 1-2 digits

                        match = re.search(pattern, text_clean)

                        if match and confidence > 0.3:
                            # Extract label based on format
                            if longitude_prefix == '' and latitude_prefix == '':
                                # Pure number/letter format
                                if match.group(1):  # Letter (latitude)
                                    label = match.group(1)
                                elif match.group(2):  # Number (longitude)
                                    label = match.group(2)
                                else:
                                    continue
                            elif longitude_prefix == '':
                                # Pure numbers OR prefixed latitude
                                if match.group(3):  # Pure number
                                    label = match.group(3)
                                elif match.group(1) and match.group(2):  # Prefixed latitude
                                    prefix = match.group(1)
                                    number = match.group(2)
                                    label = f"{prefix}{number}"
                                else:
                                    continue
                            elif latitude_prefix == '':
                                # Prefixed longitude OR pure letter
                                if match.group(3):  # Pure letter
                                    label = match.group(3)
                                elif match.group(1) and match.group(2):  # Prefixed longitude
                                    prefix = match.group(1)
                                    number = match.group(2)
                                    label = f"{prefix}{number}"
                                else:
                                    continue
                            else:
                                # Standard prefixed format
                                prefix = match.group(1)
                                number = match.group(2)
                                label = f"{prefix}{number}"

                            # Calculate position in enlarged image
                            bbox_array = np.array(bbox)
                            center_x_enlarged = int(np.mean(bbox_array[:, 0]))
                            center_y_enlarged = int(np.mean(bbox_array[:, 1]))

                            # Convert back to original image coordinates
                            center_x_roi = center_x_enlarged // enlarge_factor
                            center_y_roi = center_y_enlarged // enlarge_factor

                            # Convert to full image coordinates
                            center_x = x1 + center_x_roi
                            center_y = y1 + center_y_roi

                            labeled_positions.append((label, center_x, center_y))
                            log.info(f"Found label '{label}' in {margin_name} margin at ({center_x}, {center_y}) conf={confidence:.2f}")

                    except Exception as e:
                        log.debug(f"Error processing detection in {margin_name}: {e}")

            log.info(f"Detected {len(labeled_positions)} grid labels via margin OCR")
            return labeled_positions

        except Exception as e:
            log.error(f"Margin OCR scanning failed: {e}")
            return []

    def find_line_for_position(
        self,
        label_x: int,
        label_y: int,
        lines: List[np.ndarray],
        is_vertical: bool,
        tolerance: int = 150
    ) -> Optional[np.ndarray]:
        """
        Find the grid line that corresponds to a labeled position.

        Args:
            label_x, label_y: Label position coordinates
            lines: List of detected lines
            is_vertical: True if looking for vertical line, False for horizontal
            tolerance: Maximum distance from line to label position (increased for labels in margins)

        Returns:
            The matching line or None
        """
        best_line = None
        min_distance = float('inf')

        for line in lines:
            x1, y1, x2, y2 = line

            # Calculate distance from label position to line
            if is_vertical:
                # For vertical lines, check x-distance
                distance = abs(label_x - x1)
            else:
                # For horizontal lines, check y-distance
                distance = abs(label_y - y1)

            if distance < min_distance and distance <= tolerance:
                min_distance = distance
                best_line = line

        if best_line is not None:
            log.debug(f"Matched label at ({label_x}, {label_y}) to line with distance {min_distance}")

        return best_line

    def deduplicate_labeled_positions(
        self,
        labeled_positions: List[Tuple[str, int, int]],
        distance_threshold: int = 5
    ) -> List[Tuple[str, int, int]]:
        """
        Remove duplicate labels that are within distance_threshold of each other.
        Keep the first occurrence.

        Args:
            labeled_positions: List of (label, x, y) tuples
            distance_threshold: Maximum distance in pixels to consider as duplicate (default: 5px)

        Returns:
            Deduplicated list of (label, x, y) tuples
        """
        if not labeled_positions:
            return []

        deduplicated = []
        seen_positions = []

        for label, x, y in labeled_positions:
            # Check if this position is too close to any seen position with same label
            is_duplicate = False
            for seen_label, seen_x, seen_y in seen_positions:
                if label == seen_label:
                    distance = np.sqrt((x - seen_x)**2 + (y - seen_y)**2)
                    if distance < distance_threshold:
                        is_duplicate = True
                        log.debug(f"Removing duplicate label '{label}' at ({x}, {y}) - {distance:.1f}px from ({seen_x}, {seen_y})")
                        break

            if not is_duplicate:
                deduplicated.append((label, x, y))
                seen_positions.append((label, x, y))

        if len(labeled_positions) != len(deduplicated):
            log.info(f"Deduplication: {len(labeled_positions)} → {len(deduplicated)} labels ({len(labeled_positions) - len(deduplicated)} duplicates removed)")

        return deduplicated

    def validate_ocr_label(
        self,
        label: str,
        is_multi_characters: bool,
        longitude_prefix: str = 'X',
        latitude_prefix: str = 'Y'
    ) -> str:
        """
        Fix common OCR misreads for grid labels.

        Args:
            label: OCR-detected label
            is_multi_characters: True for X1/Y1 format, False for 1/A format
            longitude_prefix: Prefix for longitude axis
            latitude_prefix: Prefix for latitude axis

        Returns:
            Corrected label
        """
        if not is_multi_characters:
            # Single-character format: validate pure numbers and letters
            # Common misreads for numbers: "T" or "I" instead of "1"
            if label in ['T', 'I']:
                log.info(f"OCR validation: correcting '{label}' → '1'")
                return '1'
            # "O" instead of "0"
            if label == 'O':
                log.info(f"OCR validation: correcting '{label}' → '0'")
                return '0'
            # "B" could be "8" in some fonts
            # "S" could be "5" in some fonts
            # But these are more ambiguous, so skip for now
        else:
            # Multi-character format: validate prefixed labels
            # Check if label starts with expected prefix
            if not label.startswith(longitude_prefix.upper()) and not label.startswith(latitude_prefix.upper()):
                # Try to fix common prefix misreads
                if label.startswith('T') and longitude_prefix.upper() == 'X':
                    # "TX1" should be "X1" - OCR misread X as T
                    corrected = longitude_prefix.upper() + label[1:]
                    log.info(f"OCR validation: correcting '{label}' → '{corrected}'")
                    return corrected

        return label

    def process_floor_plan(
        self,
        image: np.ndarray,
        is_multi_characters: bool = True,
        has_circle: bool = True,
        longitude_prefix: str = 'X',
        latitude_prefix: str = 'Y',
        longitude_start: Optional[Dict] = None,
        latitude_start: Optional[Dict] = None,
        min_line_length: int = 100,
        max_line_gap: int = 10,
        auto_crop: bool = True,
        min_area_ratio: float = 0.15,
        min_line_length_ratio: float = 0.6
    ) -> Tuple[List[GridLine], Optional[Tuple[int, int, int, int]]]:
        """
        Process floor plan image and extract grid lines with coordinates.

        RESTRUCTURED APPROACH: Explicit case handling based on label format.

        Args:
            image: Input floor plan image
            is_multi_characters: True for X1/Y1 format, False for 1/A format
            has_circle: True if labels are in circles, False for plain text
            longitude_prefix: Prefix for longitude axis ('X', 'M', '' for single char)
            latitude_prefix: Prefix for latitude axis ('Y', 'N', '' for single char)
            longitude_start: Optional reference for longitude detection
                {
                    'label': 'X1',  # Label at this position
                    'x': int,        # X coordinate (from left-bottom corner)
                    'y': int         # Y coordinate (from left-bottom corner)
                }
            latitude_start: Optional reference for latitude detection
            min_line_length: Minimum line length in pixels
            max_line_gap: Maximum gap between line segments
            auto_crop: Automatically detect and crop to largest rectangle
            min_area_ratio: Minimum area ratio for rectangle detection
            min_line_length_ratio: Minimum length ratio for grid line filtering

        Returns:
            Tuple of (List of GridLine objects, crop_rect)
            crop_rect is None if auto_crop=False or no rectangle found
        """
        log.info(f"Processing floor plan for grid detection...")
        log.info(f"  Format: {'Multi-character' if is_multi_characters else 'Single character'} "
                f"({'with circles' if has_circle else 'plain text'})")
        log.info(f"  Prefixes: longitude='{longitude_prefix}', latitude='{latitude_prefix}'")

        crop_rect = None
        working_image = image

        # Step 1: Crop floor plan FIRST to reduce noise
        if auto_crop:
            log.info("Step 1: Detecting and cropping floor plan region...")
            cropped_image, crop_rect = ImageProcessor.find_and_crop_floor_plan(
                image,
                min_area_ratio=min_area_ratio,
                padding=10
            )
            working_image = cropped_image
            log.info(f"Cropped floor plan: {cropped_image.shape[:2]} (from {image.shape[:2]})")
        else:
            working_image = image

        # Step 2: Detect grid labels using explicit case handling
        log.info("Step 2: Detecting grid labels...")
        log.info(f"Detection strategy: Case {'1' if is_multi_characters and has_circle else '2' if is_multi_characters else '3' if has_circle else '4'}")

        labeled_positions = []
        positions_from_circles = False  # Track source of positions for coordinate adjustment

        # Step 2.1: Guided detection using reference coordinates (if provided)
        # This helps narrow down the search and improve accuracy
        reference_longitude_y = None  # Y coordinate where longitude labels should be
        reference_latitude_x = None   # X coordinate where latitude labels should be

        if longitude_start or latitude_start:
            log.info("Step 2.1: Using guided detection with reference coordinates...")

            if longitude_start:
                log.info(f"  Longitude reference: label='{longitude_start.get('label')}' at ({longitude_start.get('x')}, {longitude_start.get('y')})")
                # OCR around the longitude reference coordinate to find the start label
                ref_x = longitude_start.get('x')
                ref_y = longitude_start.get('y')
                expected_label = longitude_start.get('label')

                # Search in a small region around the reference coordinate
                search_radius = 100  # pixels
                roi_x1 = max(0, ref_x - search_radius)
                roi_x2 = min(image.shape[1], ref_x + search_radius)
                roi_y1 = max(0, ref_y - search_radius)
                roi_y2 = min(image.shape[0], ref_y + search_radius)

                roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

                # Try circle detection first if has_circle=True
                if has_circle:
                    circles = self.detect_label_circles(roi, min_radius=15, max_radius=50)
                    for x, y, radius in circles:
                        label = self.extract_label_from_circle(
                            roi, x, y, radius, padding=10,
                            longitude_prefix=longitude_prefix,
                            latitude_prefix=latitude_prefix
                        )
                        if label and (not expected_label or label == expected_label):
                            # Found the reference label - use its Y coordinate as reference
                            abs_x = roi_x1 + x
                            abs_y = roi_y1 + y
                            reference_longitude_y = abs_y
                            labeled_positions.append((label, abs_x, abs_y))
                            log.info(f"  ✓ Found longitude reference '{label}' at ({abs_x}, {abs_y})")
                            break

                # If not found via circles, try OCR
                if reference_longitude_y is None:
                    # Enlarge ROI for better OCR
                    h, w = roi.shape[:2]
                    enlarged_roi = cv2.resize(roi, (w * 6, h * 6), interpolation=cv2.INTER_CUBIC)

                    try:
                        result = self.ocr.ocr(enlarged_roi, det=True, rec=True, cls=True)
                        if result and result[0]:
                            import re
                            for detection in result[0]:
                                if isinstance(detection, (list, tuple)) and len(detection) == 2:
                                    bbox, text_info = detection
                                    if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                                        text, confidence = text_info
                                        text_clean = str(text).strip().upper()

                                        # Match expected label pattern
                                        if not is_multi_characters:
                                            # Single-char: match pure number
                                            match = re.search(r'(\d+)', text_clean)
                                        else:
                                            # Multi-char: match prefixed label
                                            pattern = rf'({longitude_prefix.upper()})\s*(\d+)'
                                            match = re.search(pattern, text_clean)

                                        if match and confidence > 0.3:
                                            if not is_multi_characters:
                                                label = match.group(1)
                                            else:
                                                label = f"{longitude_prefix.upper()}{match.group(2)}"

                                            if not expected_label or label == expected_label:
                                                # Calculate position in original image
                                                bbox_array = np.array(bbox)
                                                center_x_enlarged = int(np.mean(bbox_array[:, 0]))
                                                center_y_enlarged = int(np.mean(bbox_array[:, 1]))
                                                center_x_roi = center_x_enlarged // 6
                                                center_y_roi = center_y_enlarged // 6
                                                abs_x = roi_x1 + center_x_roi
                                                abs_y = roi_y1 + center_y_roi

                                                reference_longitude_y = abs_y
                                                labeled_positions.append((label, abs_x, abs_y))
                                                log.info(f"  ✓ Found longitude reference '{label}' at ({abs_x}, {abs_y}) via OCR")
                                                break
                    except Exception as e:
                        log.debug(f"  OCR failed for longitude reference: {e}")

            if latitude_start:
                log.info(f"  Latitude reference: label='{latitude_start.get('label')}' at ({latitude_start.get('x')}, {latitude_start.get('y')})")
                # OCR around the latitude reference coordinate
                ref_x = latitude_start.get('x')
                ref_y = latitude_start.get('y')
                expected_label = latitude_start.get('label')

                search_radius = 100
                roi_x1 = max(0, ref_x - search_radius)
                roi_x2 = min(image.shape[1], ref_x + search_radius)
                roi_y1 = max(0, ref_y - search_radius)
                roi_y2 = min(image.shape[0], ref_y + search_radius)

                roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

                # Try circle detection first if has_circle=True
                if has_circle:
                    circles = self.detect_label_circles(roi, min_radius=15, max_radius=50)
                    for x, y, radius in circles:
                        label = self.extract_label_from_circle(
                            roi, x, y, radius, padding=10,
                            longitude_prefix=longitude_prefix,
                            latitude_prefix=latitude_prefix
                        )
                        if label and (not expected_label or label == expected_label):
                            abs_x = roi_x1 + x
                            abs_y = roi_y1 + y
                            reference_latitude_x = abs_x
                            labeled_positions.append((label, abs_x, abs_y))
                            log.info(f"  ✓ Found latitude reference '{label}' at ({abs_x}, {abs_y})")
                            break

                # If not found via circles, try OCR
                if reference_latitude_x is None:
                    h, w = roi.shape[:2]
                    enlarged_roi = cv2.resize(roi, (w * 6, h * 6), interpolation=cv2.INTER_CUBIC)

                    try:
                        result = self.ocr.ocr(enlarged_roi, det=True, rec=True, cls=True)
                        if result and result[0]:
                            import re
                            for detection in result[0]:
                                if isinstance(detection, (list, tuple)) and len(detection) == 2:
                                    bbox, text_info = detection
                                    if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                                        text, confidence = text_info
                                        text_clean = str(text).strip().upper()

                                        # Match expected label pattern
                                        if not is_multi_characters:
                                            # Single-char: match pure letter
                                            match = re.search(r'([A-Z])', text_clean)
                                        else:
                                            # Multi-char: match prefixed label
                                            pattern = rf'({latitude_prefix.upper()})\s*(\d+)'
                                            match = re.search(pattern, text_clean)

                                        if match and confidence > 0.3:
                                            if not is_multi_characters:
                                                label = match.group(1)
                                            else:
                                                label = f"{latitude_prefix.upper()}{match.group(2)}"

                                            if not expected_label or label == expected_label:
                                                bbox_array = np.array(bbox)
                                                center_x_enlarged = int(np.mean(bbox_array[:, 0]))
                                                center_y_enlarged = int(np.mean(bbox_array[:, 1]))
                                                center_x_roi = center_x_enlarged // 6
                                                center_y_roi = center_y_enlarged // 6
                                                abs_x = roi_x1 + center_x_roi
                                                abs_y = roi_y1 + center_y_roi

                                                reference_latitude_x = abs_x
                                                labeled_positions.append((label, abs_x, abs_y))
                                                log.info(f"  ✓ Found latitude reference '{label}' at ({abs_x}, {abs_y}) via OCR")
                                                break
                    except Exception as e:
                        log.debug(f"  OCR failed for latitude reference: {e}")

            if reference_longitude_y or reference_latitude_x:
                log.info(f"  Reference coordinates established: longitude_y={reference_longitude_y}, latitude_x={reference_latitude_x}")
                log.info(f"  Will use these as alignment guides for detecting other labels")

        # Step 2.2: Main detection strategy (with reference guidance if available)

        # CASE 3 & 4: Single-character format (is_multi_characters=False)
        # Numbers for longitude, letters for latitude
        if not is_multi_characters:
            if has_circle:
                # CASE 3: Single-character WITH circles - Circle detection FIRST
                log.info("CASE 3: Single-character with circles - prioritizing circle detection...")
                circles = self.detect_label_circles(image, min_radius=15, max_radius=50)
                log.info(f"Detected {len(circles)} potential label circles")
            else:
                # CASE 4: Single-character plain text - Skip circle detection
                log.info("CASE 4: Single-character plain text - skipping circle detection...")
                circles = []

            if has_circle and circles:
                # Filter circles by edge proximity - grid labels should be near bottom/left edges
                # This prevents detecting circles from interior content (dimensions, room labels, etc.)
                img_height, img_width = image.shape[:2]
                edge_margin_bottom = 300  # Only consider circles within 300px of bottom edge
                edge_margin_left = 700    # Outer boundary for left edge
                min_distance_from_left = 400  # Inner boundary - reject if TOO close (dimension markers)

                filtered_circles = []
                for x, y, radius in circles:
                    # Numbers should be near bottom edge
                    near_bottom = y > img_height - edge_margin_bottom
                    # Letters should be near left edge (two-stage filter)
                    near_left = x < edge_margin_left
                    too_close_to_left = x < min_distance_from_left

                    # CRITICAL: Reject circles too close to left edge FIRST (dimension markers)
                    # This must be checked before other conditions to prevent false positives
                    if too_close_to_left:
                        log.debug(f"Rejecting circle at ({x}, {y}) - too close to left edge (< 400px, likely dimension marker)")
                        continue

                    # Accept if near bottom OR near left (but not too close)
                    if near_bottom:
                        filtered_circles.append((x, y, radius))
                        log.debug(f"Keeping bottom edge circle at ({x}, {y})")
                    elif near_left:  # Already checked not too_close_to_left above
                        filtered_circles.append((x, y, radius))
                        log.debug(f"Keeping left edge circle at ({x}, {y}) - in safe zone (400-700px)")
                    else:
                        log.debug(f"Skipping interior circle at ({x}, {y}) - too far from edges")

                log.info(f"Edge proximity filter: {len(circles)} → {len(filtered_circles)} circles (two-stage left filter: 400-700px)")

                for x, y, radius in filtered_circles:
                    label = self.extract_label_from_circle(
                        image, x, y, radius, padding=10,
                        longitude_prefix=longitude_prefix,
                        latitude_prefix=latitude_prefix
                    )
                    if label:
                        labeled_positions.append((label, x, y))
                        log.info(f"Circle detection extracted: '{label}' at ({x}, {y})")
                    else:
                        log.warning(f"Failed to extract label from circle at ({x}, {y}) with radius {radius}")

                if labeled_positions:
                    # Count what we found
                    num_numbers = sum(1 for label, _, _ in labeled_positions if label.isdigit())
                    num_letters = sum(1 for label, _, _ in labeled_positions if label.isalpha())
                    log.info(f"Circle detection: extracted {len(labeled_positions)} labels (numbers:{num_numbers}, letters:{num_letters})")

                    # If we found very few labels (< 8), try OCR as supplement
                    MIN_EXPECTED_LABELS = 8
                    if len(labeled_positions) < MIN_EXPECTED_LABELS:
                        log.warning(f"Circle detection found only {len(labeled_positions)} labels (< {MIN_EXPECTED_LABELS})")
                        log.info("Will try OCR fallback to supplement circle detection...")
                        positions_from_circles = False  # Allow OCR fallback
                    else:
                        positions_from_circles = True
                        log.info(f"Circle detection successful with {len(labeled_positions)} labels")

        # CASE 1 & 2: Multi-character format (is_multi_characters=True)
        # Use margin OCR as primary strategy
        else:  # is_multi_characters == True
            if has_circle:
                # CASE 1: Multi-character with circles - Margin OCR first, circle fallback
                log.info("CASE 1: Multi-character with circles - using margin OCR primary strategy...")
            else:
                # CASE 2: Multi-character plain text - Margin OCR only
                log.info("CASE 2: Multi-character plain text - using margin OCR only...")

        # Margin OCR fallback/primary (for both single-char and multi-char cases)
        # For single-char: supplement circle detection OR main strategy if plain text
        # For multi-char: primary strategy
        circle_labeled_positions = labeled_positions.copy()  # Save circle results

        should_try_ocr = (
            not positions_from_circles or  # Circle detection didn't find enough labels
            not has_circle                 # Plain text format - no circles expected
        )

        if should_try_ocr:
            log.info("Trying margin-based OCR...")
            crop_height, crop_width = working_image.shape[:2]

            # Determine margin sizes based on format
            if not is_multi_characters:
                # Single-character: SMALLER margins to avoid interior content
                # Numbers at bottom: narrow margin (150px) since they're typically in a single row
                # Letters at left: wide margin (600px) to catch letters spread vertically
                x_margin_height = min(crop_height, 150)  # Bottom strip up to 150px for numbers
                y_margin_width = min(crop_width, 600)    # Left strip up to 600px for letters (very wide)
                log.info(f"Single-character format: using narrow bottom margin (150px) and wide left margin (600px)")
            else:
                # Multi-character: LARGER margins (labels may be further from edges)
                x_margin_height = min(crop_height, 600)  # Bottom strip up to 600px
                y_margin_width = min(crop_width, 600)    # Left strip up to 600px
                log.info(f"Multi-character format: using larger margins (600px both)")


            log.info(f"Scanning bottom edge: {crop_width}px wide × {x_margin_height}px tall")
            log.info(f"Scanning left edge: {y_margin_width}px wide × {crop_height}px tall")

            ocr_labeled_positions = self.detect_grid_labels_by_margin_ocr(
                working_image,     # Use CROPPED floor plan (less noise)
                x_margin_height=x_margin_height,  # Bottom strip height
                y_margin_width=y_margin_width,    # Left strip width
                enlarge_factor=6,   # 6x enlargement - balance between quality and performance
                longitude_prefix=longitude_prefix,
                latitude_prefix=latitude_prefix
            )

            # Adjust OCR positions back to full image coordinates if cropped
            if crop_rect:
                offset_x, offset_y = crop_rect[0], crop_rect[1]
                ocr_labeled_positions = [
                    (label, x + offset_x, y + offset_y)
                    for label, x, y in ocr_labeled_positions
                ]
                log.info(f"Adjusted {len(ocr_labeled_positions)} OCR label positions for crop offset ({offset_x}, {offset_y})")

            # Merge circle results + OCR results (if we have both)
            if circle_labeled_positions and ocr_labeled_positions:
                log.info(f"Merging circle labels ({len(circle_labeled_positions)}) + OCR labels ({len(ocr_labeled_positions)})")
                # Remove duplicates by label (prefer circle detection)
                circle_labels = {label for label, _, _ in circle_labeled_positions}
                merged = circle_labeled_positions.copy()
                for label, x, y in ocr_labeled_positions:
                    if label not in circle_labels:
                        merged.append((label, x, y))
                        log.info(f"Added OCR label '{label}' at ({x}, {y})")
                labeled_positions = merged
                log.info(f"Merged result: {len(labeled_positions)} total labels")
            elif ocr_labeled_positions:
                # Only OCR results
                labeled_positions = ocr_labeled_positions
            # else: keep circle_labeled_positions

        # Count labels based on format
        if not is_multi_characters:
            # Single-character: separate by digit vs alpha (but don't assume which is which axis)
            digit_count = sum(1 for label, _, _ in labeled_positions if label.isdigit())
            alpha_count = sum(1 for label, _, _ in labeled_positions if label.isalpha())
            log.info(f"Margin OCR: {len(labeled_positions)} labels (digits:{digit_count}, letters:{alpha_count})")
        else:
            # Multi-character: prefixed labels
            long_labels_found = sum(1 for label, _, _ in labeled_positions if label.startswith(longitude_prefix.upper()))
            lat_labels_found = sum(1 for label, _, _ in labeled_positions if label.startswith(latitude_prefix.upper()))
            log.info(f"Margin OCR: {len(labeled_positions)} labels ({longitude_prefix}:{long_labels_found}, {latitude_prefix}:{lat_labels_found})")

        # Step 2.5: Circle fallback for multi-character format with circles
        # Only try circle fallback if:
        # - Multi-character format AND has_circle=True AND margin OCR found insufficient labels
        if is_multi_characters and has_circle:
            MIN_EXPECTED_LABELS = 10  # Expect at least 10 labels total
            if len(labeled_positions) < MIN_EXPECTED_LABELS:
                log.info(f"Step 2.5: Margin OCR found only {len(labeled_positions)} labels (< {MIN_EXPECTED_LABELS})")
                log.info("Trying circle-assisted OCR as fallback...")
                circles = self.detect_label_circles(image, min_radius=15, max_radius=50)
                log.info(f"Detected {len(circles)} potential label circles")

                if circles:
                    circle_labels = []
                    for x, y, radius in circles:
                        label = self.extract_label_from_circle(
                            image, x, y, radius, padding=10,
                            longitude_prefix=longitude_prefix,
                            latitude_prefix=latitude_prefix
                        )
                        if label:
                            circle_labels.append((label, x, y))

                    log.info(f"Circle-assisted OCR: {len(circle_labels)} labels")

                    # Use circle-assisted labels if they are more than margin labels
                    if len(circle_labels) > len(labeled_positions):
                        log.info(f"Using circle-assisted labels ({len(circle_labels)} > {len(labeled_positions)})")
                        labeled_positions = circle_labels
                        positions_from_circles = True  # Mark as from circles
                    else:
                        log.info(f"Keeping margin labels ({len(labeled_positions)} >= {len(circle_labels)})")

        if labeled_positions:
            # Count labels based on format
            if not is_multi_characters:
                # Single-character: separate by digit vs alpha (alignment filtering determines axis)
                digit_count = sum(1 for label, _, _ in labeled_positions if label.isdigit())
                alpha_count = sum(1 for label, _, _ in labeled_positions if label.isalpha())
                log.info(f"Final: {len(labeled_positions)} grid labels (digits:{digit_count}, letters:{alpha_count})")
            else:
                # Multi-character: prefixed labels
                long_labels_found = sum(1 for label, _, _ in labeled_positions if label.startswith(longitude_prefix.upper()))
                lat_labels_found = sum(1 for label, _, _ in labeled_positions if label.startswith(latitude_prefix.upper()))
                log.info(f"Final: {len(labeled_positions)} grid labels ({longitude_prefix}:{long_labels_found}, {latitude_prefix}:{lat_labels_found})")

        # Step 2.5: Filter labeled positions by alignment rules
        # This significantly reduces false positives
        if labeled_positions:
            log.info(f"[Alignment Filtering] Applying alignment rules to {len(labeled_positions)} detected labels...")

            # If reference coordinates are available, use them for more accurate filtering
            if reference_longitude_y is not None or reference_latitude_x is not None:
                log.info(f"[Guided Alignment] Using reference coordinates for filtering")

                # Filter longitude labels by reference Y coordinate
                if reference_longitude_y is not None:
                    tolerance = 50  # Allow 50px deviation from reference
                    if not is_multi_characters:
                        # Single-char: filter by digit
                        long_labels = [(label, x, y) for label, x, y in labeled_positions
                                      if label.isdigit() and abs(y - reference_longitude_y) <= tolerance]
                    else:
                        # Multi-char: filter by prefix
                        long_labels = [(label, x, y) for label, x, y in labeled_positions
                                      if label.startswith(longitude_prefix.upper()) and abs(y - reference_longitude_y) <= tolerance]

                    log.info(f"  Longitude: {len(long_labels)} labels within {tolerance}px of Y={reference_longitude_y}")
                else:
                    # No reference, separate normally
                    if not is_multi_characters:
                        long_labels = [(label, x, y) for label, x, y in labeled_positions if label.isdigit()]
                    else:
                        long_labels = [(label, x, y) for label, x, y in labeled_positions if label.startswith(longitude_prefix.upper())]

                # Filter latitude labels by reference X coordinate
                if reference_latitude_x is not None:
                    tolerance = 50
                    if not is_multi_characters:
                        # Single-char: filter by alpha
                        lat_labels = [(label, x, y) for label, x, y in labeled_positions
                                     if label.isalpha() and abs(x - reference_latitude_x) <= tolerance]
                    else:
                        # Multi-char: filter by prefix
                        lat_labels = [(label, x, y) for label, x, y in labeled_positions
                                     if label.startswith(latitude_prefix.upper()) and abs(x - reference_latitude_x) <= tolerance]

                    log.info(f"  Latitude: {len(lat_labels)} labels within {tolerance}px of X={reference_latitude_x}")
                else:
                    # No reference, separate normally
                    if not is_multi_characters:
                        lat_labels = [(label, x, y) for label, x, y in labeled_positions if label.isalpha()]
                    else:
                        lat_labels = [(label, x, y) for label, x, y in labeled_positions if label.startswith(latitude_prefix.upper())]

                # Combine filtered labels
                labeled_positions = long_labels + lat_labels
                log.info(f"  After guided filtering: {len(labeled_positions)} labels remain")

            # Apply standard alignment filtering (with or without reference guidance)
            # Use stricter tolerance for single-character format (should be perfectly aligned)
            if not is_multi_characters:
                alignment_tolerance = 10  # Stricter for single-char format (circled labels are perfectly aligned)
            else:
                alignment_tolerance = 20  # More lenient for multi-char format

            labeled_positions = self.filter_labeled_positions_by_alignment(
                labeled_positions,
                alignment_tolerance=alignment_tolerance,
                longitude_prefix=longitude_prefix,
                latitude_prefix=latitude_prefix
            )
            if not labeled_positions:
                log.warning("No labels passed alignment filtering!")

            # Apply deduplication after alignment filtering
            if labeled_positions:
                labeled_positions = self.deduplicate_labeled_positions(labeled_positions, distance_threshold=5)

        if not labeled_positions:
            log.warning("No grid labels detected via OCR, falling back to line detection")
            return self._fallback_line_detection(
                working_image, min_line_length, max_line_gap,
                min_line_length_ratio, crop_rect
            )

        log.info(f"Found {len(labeled_positions)} labeled positions via OCR")

        # Step 3: Use OCR text positions directly to create normalized grid lines
        # No line detection needed - use the text positions as the authoritative source
        log.info("Creating grid lines from OCR text positions...")

        offset_x = crop_rect[0] if crop_rect else 0
        offset_y = crop_rect[1] if crop_rect else 0

        # Get floor plan bounds (in full image coordinates)
        floor_height, floor_width = working_image.shape[:2]

        # Separate longitude and latitude labeled positions based on format
        if not is_multi_characters:
            # Single-character: numbers for longitude, letters for latitude
            long_labeled_positions = [(label, x, y) for label, x, y in labeled_positions if label.isdigit()]
            lat_labeled_positions = [(label, x, y) for label, x, y in labeled_positions if label.isalpha()]
        else:
            # Multi-character: prefixed labels
            long_labeled_positions = [(label, x, y) for label, x, y in labeled_positions if label.startswith(longitude_prefix.upper())]
            lat_labeled_positions = [(label, x, y) for label, x, y in labeled_positions if label.startswith(latitude_prefix.upper())]

        grid_lines = []

        # Create longitude-axis lines (vertical lines)
        if long_labeled_positions:
            # Lines should span the full floor plan height
            line_start_y = offset_y
            line_end_y = offset_y + floor_height

            if not is_multi_characters:
                # Single-char: use generic term (could be digits or letters depending on convention)
                long_label_type = f"vertical ({len(long_labeled_positions)} lines)"
            else:
                long_label_type = f"{longitude_prefix}"
            log.info(f"Creating {long_label_type}-axis lines with Y range: {line_start_y} to {line_end_y}")

            for label, text_x, text_y in long_labeled_positions:
                # Use text center X coordinate as the line X position
                # text_x is already in full image coordinates, so use it directly
                line_x = text_x

                grid_lines.append(GridLine(
                    label=label,
                    start_x=line_x,
                    start_y=line_start_y,
                    end_x=line_x,
                    end_y=line_end_y,
                    angle=90.0,
                    is_vertical=True
                ))
                line_desc = f"{longitude_prefix}" if is_multi_characters else "vertical"
                log.info(f"Created {line_desc}-axis line '{label}' at x={line_x}, spanning y={line_start_y} to {line_end_y}")

        # Create latitude-axis lines (horizontal lines)
        if lat_labeled_positions:
            # Lines should span the full floor plan width
            line_start_x = offset_x
            line_end_x = offset_x + floor_width

            if not is_multi_characters:
                # Single-char: use generic term (could be letters or digits depending on convention)
                lat_label_type = f"horizontal ({len(lat_labeled_positions)} lines)"
            else:
                lat_label_type = f"{latitude_prefix}"
            log.info(f"Creating {lat_label_type}-axis lines with X range: {line_start_x} to {line_end_x}")

            for label, text_x, text_y in lat_labeled_positions:
                # Use text center Y coordinate as the line Y position
                # text_y is already in full image coordinates, so use it directly
                line_y = text_y

                grid_lines.append(GridLine(
                    label=label,
                    start_x=line_start_x,
                    start_y=line_y,
                    end_x=line_end_x,
                    end_y=line_y,
                    angle=0.0,
                    is_vertical=False
                ))
                line_desc = f"{latitude_prefix}" if is_multi_characters else "horizontal"
                log.info(f"Created {line_desc}-axis line '{label}' at y={line_y}, spanning x={line_start_x} to {line_end_x}")

        # Summary log with accurate terminology
        if not is_multi_characters:
            long_desc = "vertical"
            lat_desc = "horizontal"
        else:
            long_desc = f"{longitude_prefix} (vertical)"
            lat_desc = f"{latitude_prefix} (horizontal)"

        log.info(f"Created {len(grid_lines)} grid lines from OCR text positions")
        log.info(f"  - {len(long_labeled_positions)} {long_desc} lines")
        log.info(f"  - {len(lat_labeled_positions)} {lat_desc} lines")

        return grid_lines, crop_rect

    def _fallback_line_detection(
        self,
        working_image: np.ndarray,
        min_line_length: int,
        max_line_gap: int,
        min_line_length_ratio: float,
        crop_rect: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[List[GridLine], Optional[Tuple[int, int, int, int]]]:
        """
        Fallback to old line-based detection method if circle detection fails.
        """
        log.info("Using fallback line detection method...")

        binary = self.preprocess_image(working_image)
        lines = self.detect_lines(binary, min_line_length, max_line_gap)

        if not lines:
            return [], crop_rect

        filtered_lines = self.filter_main_grid_lines(
            lines,
            working_image.shape[:2],
            min_length_ratio=min_line_length_ratio,
            exclude_border=True,
            border_margin=50
        )

        if not filtered_lines:
            return [], crop_rect

        vertical_lines, horizontal_lines = self.classify_lines(filtered_lines)
        vertical_lines = self.merge_collinear_lines(vertical_lines, is_vertical=True)
        horizontal_lines = self.merge_collinear_lines(horizontal_lines, is_vertical=False)

        labeled_vertical = self.detect_grid_labels(working_image, vertical_lines, is_vertical=True)
        labeled_horizontal = self.detect_grid_labels(working_image, horizontal_lines, is_vertical=False)

        grid_lines = []
        offset_x = crop_rect[0] if crop_rect else 0
        offset_y = crop_rect[1] if crop_rect else 0

        for label, line in labeled_vertical:
            x1, y1, x2, y2 = line
            grid_lines.append(GridLine(
                label=label,
                start_x=x1 + offset_x,
                start_y=y1 + offset_y,
                end_x=x2 + offset_x,
                end_y=y2 + offset_y,
                angle=90.0,
                is_vertical=True
            ))

        for label, line in labeled_horizontal:
            x1, y1, x2, y2 = line
            grid_lines.append(GridLine(
                label=label,
                start_x=x1 + offset_x,
                start_y=y1 + offset_y,
                end_x=x2 + offset_x,
                end_y=y2 + offset_y,
                angle=0.0,
                is_vertical=False
            ))

        return grid_lines, crop_rect

    def export_to_csv(
        self,
        grid_lines: List[GridLine],
        output_path: str | Path
    ) -> Path:
        """
        Export grid lines to CSV format matching the sample.

        CSV format:
        通り芯,始点座標(X),始点座標(Y),終点座標(X),終点座標(Y)
        X1,x1,y1,x2,y2
        ...

        Args:
            grid_lines: List of GridLine objects
            output_path: Output CSV file path

        Returns:
            Path to saved CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort lines: X lines first (sorted by number), then Y lines
        def sort_key(line: GridLine):
            # Extract number from label (e.g., "X1" -> 1)
            import re
            match = re.search(r'(\d+)', line.label)
            num = int(match.group(1)) if match else 0
            # X lines first (is_vertical=True), then Y lines
            return (not line.is_vertical, num)

        sorted_lines = sorted(grid_lines, key=sort_key)

        # Create DataFrame
        data = []
        for line in sorted_lines:
            data.append({
                '通り芯': line.label,
                '始点座標(X)': line.start_x,
                '始点座標(Y)': line.start_y,
                '終点座標(X)': line.end_x,
                '終点座標(Y)': line.end_y,
            })

        df = pd.DataFrame(data)

        # Save to CSV with UTF-8 BOM for Japanese characters
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        log.info(f"Exported {len(grid_lines)} grid lines to: {output_path}")
        return output_path

    def _save_circle_roi_debug(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        radius: int,
        idx: int,
        output_dir: str = "output"
    ) -> None:
        """
        Save ROI of a single circle for debugging.

        Args:
            image: Input image
            x, y: Circle center
            radius: Circle radius
            idx: Circle index
            output_dir: Output directory
        """
        height, width = image.shape[:2]
        padding = 15

        # Extract ROI
        roi_x1 = max(0, x - radius - padding)
        roi_x2 = min(width, x + radius + padding)
        roi_y1 = max(0, y - radius - padding)
        roi_y2 = min(height, y + radius + padding)

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size > 0:
            output_path = Path(output_dir) / f"circle_roi_{idx}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), roi)
            log.debug(f"Saved circle ROI {idx} to: {output_path}")

    def _save_circles_debug(
        self,
        image: np.ndarray,
        circles: List[Tuple[int, int, int]],
        output_dir: str = "output"
    ) -> None:
        """
        Save debug visualization of detected circles.

        Args:
            image: Input image
            circles: List of (x, y, radius) tuples
            output_dir: Output directory
        """
        debug_image = image.copy()

        for idx, (x, y, r) in enumerate(circles):
            # Draw circle
            cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
            # Draw center
            cv2.circle(debug_image, (x, y), 3, (0, 0, 255), -1)
            # Draw label
            cv2.putText(
                debug_image,
                f"#{idx+1}",
                (x - 20, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        output_path = Path(output_dir) / "detected_circles_debug.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), debug_image)
        log.info(f"Saved circles debug visualization to: {output_path}")

    def visualize_grid_lines(
        self,
        image: np.ndarray,
        grid_lines: List[GridLine],
        output_path: Optional[str | Path] = None
    ) -> np.ndarray:
        """
        Visualize detected grid lines on the image.

        Args:
            image: Original image
            grid_lines: List of GridLine objects
            output_path: Optional path to save visualization

        Returns:
            Image with grid lines drawn
        """
        # Create a copy for visualization
        vis_image = image.copy()

        # Draw lines
        for line in grid_lines:
            color = (0, 0, 255) if line.is_vertical else (255, 0, 0)  # Red for X, Blue for Y
            cv2.line(
                vis_image,
                (line.start_x, line.start_y),
                (line.end_x, line.end_y),
                color,
                2
            )

            # Draw label
            label_pos = (line.start_x + 5, line.start_y + 20)
            cv2.putText(
                vis_image,
                line.label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
            log.info(f"Saved visualization to: {output_path}")

        return vis_image
