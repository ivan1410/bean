# Bean Measurement Tool Documentation

Welcome to the documentation for the Bean Measurement Tool. This documentation is organized into the following sections:

## Documentation Files

1. [**Bean Measurement Documentation**](bean_measurement.md) - Overview, usage, parameters, and general information about the tool.

2. [**Technical Implementation**](technical_implementation.md) - In-depth explanation of the algorithms and code implementation.

## Quick Start

The Bean Measurement Tool is a computer vision application that accurately measures the width of beans in an image using a Post-it note as a size reference.

To use the tool:

1. Import the `measure_beans` function from `bean_measure.py`
2. Call it with your image path and optional parameters
3. Get back an annotated image and a list of bean measurements

```python
from bean_measure import measure_beans

annotated_img, measurements = measure_beans("path/to/your/image.jpg")
```

## Example

![Bean Measurement Example](../output_annotated.jpg)

For more detailed information, please refer to the specific documentation files. 