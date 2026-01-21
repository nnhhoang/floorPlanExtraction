"""
Auto-detection module for floor plan grid configuration.

Automatically detects:
- has_circle: Whether labels are in circles
- is_multi_characters: Whether labels are X1/Y1 format or single char (1/A)
- longitude_prefix: Prefix for vertical axis labels (X, M, or empty)
- latitude_prefix: Prefix for horizontal axis labels (Y, N, or empty)
"""
import cv2
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.utils.logger import log
from src.utils.ocr_factory import OCRFactory


@dataclass
class AutoConfig:
    """Configuration detected from floor plan image."""
    has_circle: bool
    is_multi_characters: bool
    longitude_prefix: str  # 'X', 'M', '' etc.
    latitude_prefix: str   # 'Y', 'N', '' etc.
    confidence: float      # Detection confidence (0.0 - 1.0)
    detection_details: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        format_type = "Multi-char" if self.is_multi_characters else "Single-char"
        circle_type = "with circles" if self.has_circle else "plain text"
        prefixes = f"longitude='{self.longitude_prefix}', latitude='{self.latitude_prefix}'"
        return f"AutoConfig({format_type}, {circle_type}, {prefixes}, confidence={self.confidence:.2f})"


class AutoDetector:
    """
    Auto-detect floor plan grid label configuration.
    
    Detection flow:
    1. Scan for circles in margin regions
    2. If circles found, OCR sample labels from circles
    3. If no circles, OCR sample labels from margins directly
    4. Analyze label format to determine prefixes
    """
    
    # Thresholds
    MIN_CIRCLES_THRESHOLD = 5  # Minimum circles to consider has_circle=True
    MARGIN_RATIO = 0.15  # 15% of image dimension for margin regions
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize AutoDetector.

        Args:
            use_gpu: Use GPU for OCR if available (default: True)
        """
        # Use shared OCR instance via factory for better performance
        # EasyOCR initialization is expensive (~2-5s, ~500MB+ memory)
        self.ocr = OCRFactory.get_reader(use_gpu=use_gpu)
        log.info("AutoDetector initialized with shared EasyOCR instance")
    
    def detect_configuration(self, image: np.ndarray) -> AutoConfig:
        """
        Detect floor plan configuration from image.
        
        Args:
            image: Input floor plan image (BGR)
            
        Returns:
            AutoConfig with detected settings
        """
        log.info("=" * 50)
        log.info("Starting Auto-Detection...")
        log.info("=" * 50)
        
        details = {}
        
        # Step 1: Scan for circles
        log.info("\n[1/3] Scanning for label circles...")
        circles, has_circle = self._scan_for_circles(image)
        details['circles_found'] = len(circles)
        details['has_circle'] = has_circle
        log.info(f"      Found {len(circles)} circles → has_circle={has_circle}")
        
        # Step 2: Sample OCR labels
        log.info("\n[2/3] Sampling labels via OCR...")
        if has_circle:
            sample_labels = self._sample_ocr_from_circles(image, circles)
        else:
            sample_labels = self._sample_ocr_from_margins(image)
        details['sample_labels'] = sample_labels
        log.info(f"      Sampled labels: {sample_labels}")
        
        # Step 3: Analyze label format
        log.info("\n[3/3] Analyzing label format...")
        is_multi_char, long_prefix, lat_prefix, confidence = self._analyze_label_format(sample_labels)
        details['is_multi_characters'] = is_multi_char
        details['longitude_prefix'] = long_prefix
        details['latitude_prefix'] = lat_prefix
        
        config = AutoConfig(
            has_circle=has_circle,
            is_multi_characters=is_multi_char,
            longitude_prefix=long_prefix,
            latitude_prefix=lat_prefix,
            confidence=confidence,
            detection_details=details
        )
        
        log.info("=" * 50)
        log.info(f"Auto-Detection Complete: {config}")
        log.info("=" * 50)
        
        return config
    
    def _scan_for_circles(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, np.ndarray]], bool]:
        """
        Scan margin regions for label circles.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (list of (x, y, radius, margin_img) tuples, has_circle boolean)
        """
        height, width = image.shape[:2]
        margin_h = int(height * self.MARGIN_RATIO)
        margin_w = int(width * self.MARGIN_RATIO)
        
        # Define margin regions to scan
        # Bottom margin (for longitude labels like X1, X2)
        # Left margin (for latitude labels like Y1, Y2)
        margins = {
            'bottom': image[height - margin_h:height, :],
            'left': image[:, :margin_w],
            'top': image[:margin_h, :],
            'right': image[:, width - margin_w:width],
        }
        
        all_circles = []
        
        for margin_name, margin_img in margins.items():
            circles = self._detect_circles_in_region(margin_img)
            log.debug(f"  {margin_name} margin: {len(circles)} circles")
            # Store margin_img reference with each circle for correct OCR later
            for x, y, radius in circles:
                all_circles.append((x, y, radius, margin_img))
        
        has_circle = len(all_circles) >= self.MIN_CIRCLES_THRESHOLD
        return all_circles, has_circle
    
    def _detect_circles_in_region(
        self,
        image: np.ndarray,
        min_radius: int = 15,
        max_radius: int = 50
    ) -> List[Tuple[int, int, int]]:
        """
        Detect circles in a region using contour-based detection.
        
        Args:
            image: Input image region
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            
        Returns:
            List of (x, y, radius) tuples
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        detected_circles = []
        seen_positions = set()
        
        for thresh_val in [200, 180, 160, 220]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < np.pi * min_radius**2 or area > np.pi * max_radius**2:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.65:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    x, y, radius = int(x), int(y), int(radius)
                    
                    if min_radius <= radius <= max_radius:
                        pos_key = (x // 10, y // 10)
                        if pos_key not in seen_positions:
                            detected_circles.append((x, y, radius))
                            seen_positions.add(pos_key)
        
        return detected_circles
    
    def _sample_ocr_from_circles(
        self,
        image: np.ndarray,
        circles: List[Tuple[int, int, int, np.ndarray]],
        max_samples: int = 8  # Sample more circles for better detection
    ) -> List[str]:
        """
        OCR sample labels from detected circles.
        
        Args:
            image: Full image (not used, kept for compatibility)
            circles: List of (x, y, radius, margin_img) tuples
            max_samples: Maximum number of circles to sample
            
        Returns:
            List of detected label strings
        """
        labels = []
        sampled = circles[:max_samples]
        
        for x, y, radius, margin_img in sampled:
            # Use margin_img for OCR (coordinates are relative to margin)
            label = self._extract_label_from_circle(margin_img, x, y, radius)
            if label:
                labels.append(label)
                log.info(f"      Circle at ({x}, {y}) -> OCR: '{label}'")
        
        return labels
    
    def _extract_label_from_circle(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        radius: int,
        padding: int = 10
    ) -> Optional[str]:
        """
        Extract text from a circle region.
        
        Args:
            image: Full image
            x, y: Circle center
            radius: Circle radius
            padding: Padding around circle
            
        Returns:
            Extracted text or None
        """
        height, width = image.shape[:2]
        
        roi_x1 = max(0, x - radius - padding)
        roi_x2 = min(width, x + radius + padding)
        roi_y1 = max(0, y - radius - padding)
        roi_y2 = min(height, y + radius + padding)
        
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return None
        
        # Enlarge for better OCR
        scale = 5
        h, w = roi.shape[:2]
        enlarged = cv2.resize(roi, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        
        try:
            result = self.ocr.readtext(enlarged)
            if result:
                # Join all detected texts
                full_text = ''.join([det[1] for det in result]).strip()
                return full_text if full_text else None
        except Exception as e:
            log.debug(f"OCR failed for circle at ({x}, {y}): {e}")
        
        return None
    
    def _sample_ocr_from_margins(self, image: np.ndarray) -> List[str]:
        """
        OCR sample labels from image margins (when no circles detected).
        
        Args:
            image: Input image
            
        Returns:
            List of detected label strings
        """
        height, width = image.shape[:2]
        margin_h = int(height * self.MARGIN_RATIO)
        margin_w = int(width * self.MARGIN_RATIO)
        
        labels = []
        
        # Scan bottom and left margins
        margins = {
            'bottom': image[height - margin_h:height, :],
            'left': image[:, :margin_w],
        }
        
        for margin_name, margin_img in margins.items():
            try:
                result = self.ocr.readtext(margin_img)
                for detection in result:
                    if len(detection) >= 2:
                        text = str(detection[1]).strip()
                        if text:
                            labels.append(text)
            except Exception as e:
                log.debug(f"OCR failed for {margin_name} margin: {e}")
        
        return labels
    
    def _analyze_label_format(
        self,
        labels: List[str]
    ) -> Tuple[bool, str, str, float]:
        """
        Analyze detected labels to determine format and prefixes.
        
        Args:
            labels: List of detected label strings
            
        Returns:
            Tuple of (is_multi_characters, longitude_prefix, latitude_prefix, confidence)
        """
        if not labels:
            log.warning("No labels to analyze, using defaults")
            return True, 'X', 'Y', 0.0
        
        # Patterns to detect
        patterns = {
            # Multi-char patterns: X1, Y1, M1, N1, etc.
            'multi_X': re.compile(r'^X\d{1,2}$', re.IGNORECASE),
            'multi_Y': re.compile(r'^Y\d{1,2}$', re.IGNORECASE),
            'multi_M': re.compile(r'^M\d{1,2}$', re.IGNORECASE),
            'multi_N': re.compile(r'^N\d{1,2}$', re.IGNORECASE),
            # Single-char patterns: 1, 2, A, B, etc.
            'single_num': re.compile(r'^\d{1,2}$'),
            'single_letter': re.compile(r'^[A-Z]$', re.IGNORECASE),
        }
        
        matches = {key: 0 for key in patterns}
        
        for label in labels:
            clean_label = label.strip().upper()
            for pattern_name, pattern in patterns.items():
                if pattern.match(clean_label):
                    matches[pattern_name] += 1
        
        log.info(f"      Pattern matches: {matches}")
        
        total_matches = sum(matches.values())
        if total_matches == 0:
            log.warning("No pattern matches found, using defaults")
            return True, 'X', 'Y', 0.0
        
        # Determine format
        multi_matches = matches['multi_X'] + matches['multi_Y'] + matches['multi_M'] + matches['multi_N']
        single_matches = matches['single_num'] + matches['single_letter']
        
        log.info(f"      Multi-char matches: {multi_matches}, Single-char matches: {single_matches}")
        
        # FIXED: Single-char wins if it has more matches OR if multi has 0 matches
        # Previously: is_multi_char = multi_matches >= single_matches (BUG: 0 >= 0 = True)
        if multi_matches == 0 and single_matches > 0:
            is_multi_char = False
        elif single_matches == 0 and multi_matches > 0:
            is_multi_char = True
        else:
            # Both have matches - use the one with more
            is_multi_char = multi_matches > single_matches
        
        # Determine prefixes
        if is_multi_char:
            # Check which prefix pair is most common
            xy_count = matches['multi_X'] + matches['multi_Y']
            mn_count = matches['multi_M'] + matches['multi_N']
            
            if mn_count > xy_count:
                long_prefix = 'M'
                lat_prefix = 'N'
            else:
                long_prefix = 'X'
                lat_prefix = 'Y'
        else:
            # Single-char format
            long_prefix = ''
            lat_prefix = ''
        
        # Calculate confidence
        confidence = total_matches / max(len(labels), 1)
        confidence = min(confidence, 1.0)
        
        format_type = "Multi-char" if is_multi_char else "Single-char"
        log.info(f"      Format: {format_type}, prefixes: longitude='{long_prefix}', latitude='{lat_prefix}'")
        
        return is_multi_char, long_prefix, lat_prefix, confidence
