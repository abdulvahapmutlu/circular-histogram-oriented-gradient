# Circular Histogram Oriented Gradient Feature Extraction
# Circular Histogram Oriented Gradient Feature Extraction

This repository contains Python code for extracting Circular Histogram Oriented Gradient (CHOG) features from images, utilizing Local Binary Pattern (LBP) and Histogram of Oriented Gradients (HOG) techniques. This feature extraction is particularly useful in computer vision applications, enabling enhanced image classification and pattern recognition.

## Features

- **LBP Feature Extraction**: Captures local texture information by analyzing the binary patterns in an image.
- **HOG Feature Extraction**: Analyzes the gradient orientation and edge structure within the image.
- **Circular Image Analysis**: Extracts features from rotated versions of the image to achieve orientation invariance.
- **Feature Normalization**: Scales the feature values to a standard range using MinMaxScaler.
- **Feature Selection**: Uses ANOVA F-test to select the most significant features for classification.

## Installation

To use this code, ensure you have the following dependencies installed:

- `numpy`
- `opencv-python`
- `scikit-image`
- `scikit-learn`
- `scipy`

You can install these using pip:

```bash
pip install numpy opencv-python scikit-image scikit-learn scipy
```

## Usage

1. Place your images in the working directory. Ensure each image file name starts with a numeric label followed by an underscore (e.g., `1_image.jpg`).
2. Run the script to extract features from the images:

```python
python main_circular_hog.py
```

3. The script will read each image, apply LBP and HOG feature extraction, normalize the features, perform feature selection, and output the combined features and labels.

## Example

Suppose you have the following image files in your directory:

```
1_image1.jpg
2_image2.jpg
3_image3.jpg
...
```

After running the script, it will process each image, extract and normalize the features, and save the selected features and labels for further use.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Your contributions are greatly appreciated!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
