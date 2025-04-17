# Bean Measurement Tool Documentation

## Overview

The Bean Measurement Tool is a computer vision application that accurately measures the width of beans in an image using a Post-it note as a size reference. The script uses color-based segmentation, contour detection, and perspective transformation to achieve accurate measurements.

## Dependencies

- OpenCV (cv2)
- NumPy

## Installation

```bash
pip install opencv-python numpy
```

## How It Works

The script follows these key steps:

1. **Image Loading**: Reads the input image containing beans and a Post-it note.
2. **Color Space Conversion**: Converts the image to HSV color space for better color segmentation.
3. **Post-it Detection**: 
   - Identifies the Post-it note by color thresholding in HSV space
   - Finds the largest quadrilateral contour (the Post-it)
   - Orders its corners (top-left, top-right, bottom-right, bottom-left)
4. **Perspective Transformation**:
   - Uses the Post-it corners to create a transformation matrix
   - This establishes a metric conversion where 1 pixel = 1 mm (based on known Post-it size)
5. **Bean Detection**:
   - Identifies beans using color thresholding in HSV space
   - Applies morphological operations to clean up the mask
   - Finds and filters bean contours
6. **Width Measurement**:
   - Transforms each bean contour to the metric space
   - Calculates the minimum width using minAreaRect
   - Converts pixel measurements to millimeters
7. **Result Visualization**:
   - Annotates the original image with bean IDs and width measurements
   - Outputs an annotated image and measurement data

## Usage

### Basic Usage

```python
from bean_measure import measure_beans

# Run measurement with default parameters
image_path = "path/to/image.jpg"
annotated_img, bean_measurements = measure_beans(image_path)
```

### Advanced Usage

```python
from bean_measure import measure_beans

# With custom parameters
annotated_img, bean_measurements = measure_beans(
    image_path="path/to/image.jpg",
    postit_side_mm=76.0,              # Post-it size in mm (standard is 76mm)
    postit_lower=(35, 0, 180),        # HSV lower threshold for Post-it
    postit_upper=(55, 80, 255),       # HSV upper threshold for Post-it
    bean_lower=(20, 100, 100),        # HSV lower threshold for beans
    bean_upper=(70, 255, 255),        # HSV upper threshold for beans
    debug=True                        # Enable debug output
)

# bean_measurements contains tuples of (bean_id, width_mm)
```

## Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | string | - | Path to the input image |
| `postit_side_mm` | float | 76.0 | Side length of Post-it note in mm |
| `postit_lower` | tuple | (20, 80, 80) | HSV lower threshold for Post-it |
| `postit_upper` | tuple | (40, 255, 255) | HSV upper threshold for Post-it |
| `bean_lower` | tuple | (35, 50, 50) | HSV lower threshold for beans |
| `bean_upper` | tuple | (90, 255, 255) | HSV upper threshold for beans |
| `debug` | bool | False | Enable/disable debug printing |

## Return Values

The `measure_beans` function returns:

1. `annotated_img`: The original image with annotations showing bean IDs and measurements
2. `results`: A list of tuples in the format `(bean_id, width_mm)`

## Image Requirements

For best results:
- Place a Post-it note (preferably yellow or another distinct color) in the same plane as the beans
- Ensure beans are clearly visible and separated from the background
- Provide good lighting with minimal shadows
- Ensure beans and Post-it are in the same focal plane

## Calibration Tips

If bean detection is not accurate:
1. Adjust the HSV thresholds for beans (`bean_lower` and `bean_upper`)
2. Try different values for the Post-it HSV range if it's not being detected properly
3. The default thresholds work best with yellow Post-its and green beans

## Limitations

- The script assumes beans and Post-it are on the same plane
- Color-based detection may be affected by lighting conditions
- Very small objects might be filtered out as noise
- Overlapping beans may be detected as a single object

## Troubleshooting

- **No Post-it detected**: Adjust the HSV range for the Post-it color
- **Beans not detected**: Modify the HSV range for bean color
- **Inaccurate measurements**: Ensure the Post-it is flat and correctly detected
- **Poor annotations**: Adjust the text positioning or font size in the code 