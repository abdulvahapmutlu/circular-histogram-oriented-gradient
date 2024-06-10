import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import scipy.io as sio

warnings.filterwarnings("ignore")


# Function to extract LBP features
def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


# Function to extract HOG features
def extract_hog_features(image, pixels_per_cell=(16, 16)):
    hog_features = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=(2, 2), visualize=False,
                       multichannel=False)
    return hog_features


# Get list of image files
files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Initialize lists to store features and labels
X = []
y = []

# Process each image file
for file in files:
    try:
        label = int(file.split('_')[0])  # Assuming the label is before an underscore in the file name
        y.append(label)

        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading image: {file}")
            continue

        # Ensure the image size matches the expected size
        image_resized = cv2.resize(image, (257, 257))

        rotations = [
            image_resized,
            cv2.rotate(image_resized, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(image_resized, cv2.ROTATE_180),
            cv2.rotate(image_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        center = 128  # Center point for 257x257 image
        features = []

        for i in range(4):
            size = 2 ** (3 + i)
            if center - size < 0 or center + size > 257:
                print(f"Error: Extraction size {size} out of bounds for image {file}")
                continue

            ex = rotations[i][center - size:center + size, center - size:center + size]
            lbp_features = extract_lbp_features(ex)
            hog_features = extract_hog_features(ex)
            features.extend(lbp_features)
            features.extend(hog_features)

        X.append(features)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Feature selection using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=256)
X_new = selector.fit_transform(X, y)

# Combine the selected features and labels
son = np.hstack((X_new, y.reshape(-1, 1)))
