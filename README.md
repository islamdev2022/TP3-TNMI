# DetectingOutlines.py

## Overview
`DetectingOutlines.py` is a Python script designed to detect and highlight outlines in images. This script utilizes image processing techniques to identify and emphasize the edges within an image.


### Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python DetectingOutlines.py
```

## Functionality
The script performs the following steps:
1. Reads the input image.
2. Converts the image to grayscale.
3. Applies a Gaussian blur to reduce noise.
4. Uses the Canny edge detection algorithm to detect edges.
5. Saves the resulting image with detected outlines to the specified output path.


## Additional Features
The script also includes a graphical user interface (GUI) that allows users to upload an image, add noise, apply filters, and perform thresholding and LOG.

### Usage
When running the program, an interface will appear that allows you to:
1. Upload an image.
2. Add noise to the image using "salt and pepper" or "Gaussian" noise.
3. Filter the noisy image using "Prewitt", "Sobel", or "Roberts" filters.
4. Apply thresholding (seuillage) to the filtered image.
5. Apply LOG to the original image 

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author
Birouk Mohammed Islam
