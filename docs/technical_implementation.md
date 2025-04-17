# Bean Measurement Tool: Technical Implementation

This document provides an in-depth explanation of the technical implementation of the bean measurement script.

## Core Functions

### `order_corners(pts)`

Orders a set of four points in the sequence: top-left, top-right, bottom-right, bottom-left.

**Implementation details:**
- First sorts points by the sum of their x and y coordinates
- This places the top-left (smallest sum) and bottom-right (largest sum) points
- Then differentiates between top-right and bottom-left by comparing y-coordinates
- Returns the ordered points as a NumPy array with float32 data type

### `measure_beans()`

Main function that processes an image and returns bean measurements.

## Algorithm Steps in Detail

### 1. Image Preprocessing

```python
image = cv2.imread(image_path)
orig = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

The image is loaded using OpenCV and converted to HSV color space. HSV (Hue, Saturation, Value) is used instead of RGB because it separates color information from intensity, making it more robust for color-based segmentation under varying lighting conditions.

### 2. Post-it Note Detection

```python
mask_postit = cv2.inRange(hsv, postit_lower, postit_upper)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask_postit = cv2.morphologyEx(mask_postit, cv2.MORPH_CLOSE, kernel)
```

- `inRange()` creates a binary mask where pixels within the specified HSV range are white (255) and others are black (0)
- Morphological closing (dilation followed by erosion) fills small holes in the mask
- The default HSV range targets yellowish colors, but can be adjusted for different colored Post-its

### 3. Post-it Contour Extraction

```python
contours_postit, _ = cv2.findContours(mask_postit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

- `findContours()` identifies the boundaries of objects in the binary mask
- `RETR_EXTERNAL` retrieves only the outermost contours
- `CHAIN_APPROX_SIMPLE` compresses horizontal, vertical, and diagonal segments to their endpoints

### 4. Post-it Shape Approximation

```python
for cnt in contours_postit:
    area = cv2.contourArea(cnt)
    if area < 100:  # skip very small
        continue
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4 and area > max_area:
        max_area = area
        square_contour = approx
```

- Contours with area < 100 pixels are ignored as noise
- `arcLength()` calculates the perimeter of the contour
- `approxPolyDP()` approximates a curve/contour with a specified precision (0.02 * perimeter)
- The largest 4-point polygon is assumed to be the Post-it note

### 5. Perspective Transformation

```python
M = cv2.getPerspectiveTransform(corners, dst_pts)
M_inv = np.linalg.inv(M)
```

- The ordered corners are used to compute a perspective transformation matrix
- The destination points form a square with dimensions equal to the Post-it's real-world size in mm
- This creates a mapping where 1 pixel = 1 mm in the warped image
- An inverse transformation matrix is calculated to map points back to the original image space

### 6. Bean Detection

```python
mask_beans = cv2.inRange(hsv, bean_lower, bean_upper)
mask_beans = cv2.bitwise_and(mask_beans, cv2.bitwise_not(mask_postit))
```

- Similar to the Post-it detection, beans are identified using HSV color thresholding
- The Post-it region is removed from the bean mask to avoid false positives
- Morphological operations clean up the mask:
  - Closing (dilate then erode) to fill holes
  - Opening (erode then dilate) to remove small noise

### 7. Measurement Methodology

```python
cnt_float = cnt.astype(np.float32)
cnt_warped = cv2.perspectiveTransform(cnt_float, M)
rect = cv2.minAreaRect(cnt_warped)
(center_warped), (w, h), angle = rect
bean_width_mm = min(w, h)
```

- Each bean contour is transformed to the metric space using the perspective transform
- In this space, pixel distances correspond to mm distances
- `minAreaRect()` finds the minimum area rotated rectangle enclosing the bean
- The smaller dimension of this rectangle is taken as the bean width

### 8. Visualization and Output

```python
center_warped_pt = np.array([[center_warped]], dtype=np.float32)
center_original = cv2.perspectiveTransform(center_warped_pt, M_inv)
cx, cy = center_original[0][0]
```

- The center of each bean in the warped space is transformed back to the original image
- Text positions are calculated using this transformed center
- Double text rendering (black outline with white text) ensures visibility

## Technical Considerations

### Data Types and Precision

- `np.float32` is used for transformation matrices and point coordinates for compatibility with OpenCV functions
- Contours are converted from their default integer type to float32 before perspective transformation

### Performance Optimizations

- `CHAIN_APPROX_SIMPLE` contour retrieval mode reduces memory usage by simplifying contours
- Small contours are filtered out early to reduce processing time
- The script processes the entire image at once rather than using sliding windows

### Error Handling

- The script checks if the image was loaded successfully
- Validates that a Post-it contour was found
- Filters out very small bean contours that might be noise

### HSV Color Spaces

The default HSV thresholds are set for:
- Yellow Post-it: H: [20-40], S: [80-255], V: [80-255]
- Green beans: H: [35-90], S: [50-255], V: [50-255]

In OpenCV, the ranges are:
- H: 0-179 (not 0-360 as in standard HSV)
- S: 0-255
- V: 0-255

## Algorithm Complexity

- Time complexity: O(n), where n is the number of pixels in the image
- Space complexity: O(n) for storing the original image, masks, and transformed coordinates 