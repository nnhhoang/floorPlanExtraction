"""Image processing utilities for floor plan extraction"""
import cv2
import numpy as np
from typing import Tuple, Optional
from src.utils.logger import log


class ImageProcessor:
    """Process and prepare images for floor plan extraction."""

    @staticmethod
    def detect_largest_rectangle(
        image: np.ndarray,
        min_area_ratio: float = 0.1,
        approx_epsilon: float = 0.02
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest rectangle (black frame) in the image.

        Args:
            image: Input image (BGR or RGB)
            min_area_ratio: Minimum area ratio relative to image size (default: 0.1 = 10%)
            approx_epsilon: Epsilon for contour approximation (default: 0.02)

        Returns:
            Tuple of (x, y, width, height) for the largest rectangle, or None if not found
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Method 1: Detect dark/black borders using thresholding
        # Invert: black borders become white
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours from binary image (dark areas)
        contours_binary, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Method 2: Traditional edge detection (as fallback)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours_edges, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Combine both methods - prioritize binary thresholding contours
        contours = list(contours_binary) + list(contours_edges)

        if not contours:
            log.warning("No contours found in image")
            return None

        # Calculate minimum area threshold
        img_height, img_width = gray.shape
        img_area = img_width * img_height
        min_area = img_area * min_area_ratio

        # Find the largest rectangular contour
        largest_rect = None
        largest_area = 0
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip if too small
            if area < min_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check aspect ratio is reasonable (not too narrow)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue

            # Calculate coverage percentage
            coverage = (w * h) / img_area

            # Approximate contour to polygon to check rectangularity
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, approx_epsilon * peri, True)

            # Check if it's approximately rectangular (4-6 corners for tolerance)
            # We allow up to 6 corners to handle slightly imperfect rectangles
            if 4 <= len(approx) <= 6:
                # Calculate rectangularity (how close to a perfect rectangle)
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0

                # Store candidate with score
                # Prefer rectangles that:
                # 1. Have high rectangularity (close to 1.0)
                # 2. Cover 75-87% of the image (typical floor plan size)
                # 3. Are not too small or too large
                score = 0
                if 0.75 < coverage < 0.87:  # Ideal coverage range (floor plan with black border)
                    score += 100
                elif 0.70 < coverage < 0.90:  # Acceptable range
                    score += 25  # Lower score for borderline cases

                if rectangularity > 0.90:  # Very rectangular
                    score += 50
                elif rectangularity > 0.80:  # Reasonably rectangular
                    score += 25

                candidates.append({
                    'rect': (x, y, w, h),
                    'area': area,
                    'coverage': coverage,
                    'rectangularity': rectangularity,
                    'corners': len(approx),
                    'score': score
                })

        # Log all candidates for debugging
        if candidates:
            log.info(f"Found {len(candidates)} candidate rectangles:")
            for i, cand in enumerate(candidates[:5]):  # Show top 5
                log.info(
                    f"  Candidate {i+1}: score={cand['score']:.0f}, "
                    f"coverage={cand['coverage']*100:.1f}%, "
                    f"rectangularity={cand['rectangularity']:.2f}, "
                    f"corners={cand['corners']}, "
                    f"size={cand['rect'][2]}x{cand['rect'][3]}"
                )

        # Sort candidates by score (descending)
        candidates.sort(key=lambda c: c['score'], reverse=True)

        # If we have good candidates, pick the best one
        if candidates and candidates[0]['score'] > 0:
            best = candidates[0]
            largest_rect = best['rect']
            log.info(
                f"Selected rectangle with score={best['score']:.0f}, "
                f"coverage={best['coverage']*100:.1f}%, "
                f"rectangularity={best['rectangularity']:.2f}, "
                f"corners={best['corners']}"
            )
        # If no good rectangle found, try bounding rect of largest contour with strict coverage limit
        elif len(candidates) == 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area >= min_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                coverage = (w * h) / img_area

                # Only use fallback if coverage is in acceptable range (70-90%)
                # This prevents capturing too much area (like entire page with text)
                if 0.70 <= coverage <= 0.90:
                    largest_rect = (x, y, w, h)
                    log.info(f"Using bounding rect of largest contour (area: {area:.0f}, coverage: {coverage*100:.1f}%)")
                else:
                    log.warning(f"Largest contour has unusual coverage {coverage*100:.1f}%, skipping")

        if largest_rect:
            x, y, w, h = largest_rect
            coverage = (w * h) / img_area * 100
            log.info(
                f"Detected largest rectangle: x={x}, y={y}, w={w}, h={h} "
                f"(covers {coverage:.1f}% of image)"
            )
        else:
            log.warning("No suitable rectangle found in image")

        return largest_rect

    @staticmethod
    def crop_to_rectangle(
        image: np.ndarray,
        rect: Tuple[int, int, int, int],
        padding: int = 0
    ) -> np.ndarray:
        """
        Crop image to specified rectangle with optional padding.

        Args:
            image: Input image
            rect: Rectangle (x, y, width, height)
            padding: Padding in pixels to add around the rectangle

        Returns:
            Cropped image
        """
        x, y, w, h = rect
        img_height, img_width = image.shape[:2]

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_width, x + w + padding)
        y2 = min(img_height, y + h + padding)

        cropped = image[y1:y2, x1:x2]

        log.info(f"Cropped image from {image.shape} to {cropped.shape}")
        return cropped

    @staticmethod
    def find_and_crop_floor_plan(
        image: np.ndarray,
        min_area_ratio: float = 0.1,
        padding: int = 10
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Detect and crop the floor plan region (largest rectangle).

        Args:
            image: Input image
            min_area_ratio: Minimum area ratio for rectangle detection
            padding: Padding around detected rectangle

        Returns:
            Tuple of (cropped_image, rectangle_coords)
            If no rectangle found, returns (original_image, None)
        """
        rect = ImageProcessor.detect_largest_rectangle(image, min_area_ratio)

        if rect is None:
            log.warning("No floor plan rectangle detected, using full image")
            return image, None

        cropped = ImageProcessor.crop_to_rectangle(image, rect, padding)
        return cropped, rect

    @staticmethod
    def validate_image_has_content(
        image: np.ndarray,
        min_edge_ratio: float = 0.01
    ) -> dict:
        """
        Validate if image contains floor plan content (not blank).

        Uses multiple techniques to detect if the image contains meaningful content:
        - Edge detection to find lines/structures
        - Histogram variance to detect non-uniform colors
        - Morphological operations to detect text/lines

        Args:
            image: Input image (BGR, RGB, or grayscale)
            min_edge_ratio: Minimum ratio of edge pixels to total pixels (default: 0.01 = 1%)

        Returns:
            Dictionary with validation results:
            {
                'has_content': bool,
                'edge_ratio': float,
                'line_ratio': float,
                'hist_variance': float,
                'confidence': float (0-1)
            }
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape
        total_pixels = height * width

        # 1. Check histogram variance (detect non-uniform colors)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_variance = float(np.var(hist))

        # 2. Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_ratio = edge_pixels / total_pixels

        # 3. Check for text/lines using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        line_pixels = np.count_nonzero(morph)
        line_ratio = line_pixels / total_pixels

        # 4. Determine if has content based on multiple criteria
        has_content = (
            edge_ratio >= min_edge_ratio and
            hist_variance > 100 and  # Not uniform color
            line_ratio > 0.005       # Has lines/structures (0.5% of pixels)
        )

        # 5. Calculate confidence score (0-1)
        # Higher edge ratio and histogram variance = higher confidence
        confidence = min(1.0, (edge_ratio / min_edge_ratio) * (hist_variance / 1000))

        log.info(
            f"Image validation: has_content={has_content}, "
            f"edge_ratio={edge_ratio:.4f}, line_ratio={line_ratio:.4f}, "
            f"hist_variance={hist_variance:.1f}, confidence={confidence:.2f}"
        )

        return {
            'has_content': bool(has_content),
            'edge_ratio': float(edge_ratio),
            'line_ratio': float(line_ratio),
            'hist_variance': float(hist_variance),
            'confidence': float(confidence)
        }

    @staticmethod
    def visualize_rectangle(
        image: np.ndarray,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> np.ndarray:
        """
        Draw rectangle on image for visualization.

        Args:
            image: Input image
            rect: Rectangle (x, y, width, height)
            color: Rectangle color (B, G, R)
            thickness: Line thickness

        Returns:
            Image with rectangle drawn
        """
        vis_image = image.copy()
        x, y, w, h = rect
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)

        # Add label
        label = f"Floor Plan: {w}x{h}"
        cv2.putText(
            vis_image,
            label,
            (x + 10, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

        return vis_image
