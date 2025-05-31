import os
import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matchPics import matchPics
from helper import plotMatches
from scipy.ndimage import rotate

resultsdir = "../results/rotTest"

"""
Q3.5
"""
if __name__ == "__main__":
    os.makedirs(resultsdir, exist_ok=True)

    # Read the image (BGR format)
    originalImg = cv2.imread("../data/cv_cover.jpg")

    nMatches = []
    # Generate angles from 0°, 10°, 20°, ... up to 360° (inclusive)
    angles = [i * 10 for i in range(37)]  # 0, 10, 20, ..., 360

    for angle in angles:
        # 1) Rotate Image
        rotImg = rotate(originalImg, angle, reshape=False)
        rotImg = rotImg.astype(np.uint8)  # Convert from float to uint8

        # 2) Compute features, descriptors, and match
        matches, locs1, locs2 = matchPics(originalImg, rotImg)

        # 3) Update histogram with the number of matches
        nMatches.append(len(matches))

        # 4) Save match visualization
        saveTo = os.path.join(resultsdir, f"rot{angle}.png")
        plotMatches(originalImg, rotImg, matches, locs1, locs2, saveTo=saveTo, showImg=True)
        
# 5) Display histogram of number of matches vs. rotation angle
plt.figure(figsize=(8, 5))
plt.bar(angles, nMatches, width=8, align='center')

plt.xticks(angles, rotation=90, fontsize=8)  # Rotate labels for readability
plt.yticks(fontsize=10)

plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.title('BRIEF Descriptor Performance Under Rotation')

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()