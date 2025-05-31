import numpy as np
import cv2
import os
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches

# Paths to the two images
left_img_path = "../data/bottle_left.jpg"
right_img_path = "../data/bottle_right.jpg"
output_path = "../results/panorama_final_bottle.jpg"

def getHomography(left, right):
    """
    Compute the homography mapping points from the right image to the left image.
    Uses matchPics to detect correspondences, converts keypoints from (row, col) to (x, y),
    and then uses computeH_ransac.
    """
    matches, locs_left, locs_right = matchPics(left, right)
    
    print(f"Number of matches found: {len(matches)}")
    
    if len(matches) < 4:
        # Debug: Display feature matching if too few matches
        plotMatches(left, right, matches, locs_left, locs_right, showImg=True)
        raise ValueError("Not enough matches to compute homography.")
    
    # Extract matched keypoints
    pts_left = locs_left[matches[:, 0], :]  # from left image
    pts_right = locs_right[matches[:, 1], :]  # from right image
    pts_left_xy = pts_left[:, [1, 0]]  # Convert to (x, y)
    pts_right_xy = pts_right[:, [1, 0]]  # Convert to (x, y)

    # Compute homography using RANSAC
    H, inliers = computeH_ransac(pts_left_xy, pts_right_xy, nSamples=50, threshold=5)  # Reduced threshold
    
    if H is None:
        raise ValueError("Homography computation failed.")
    
    # Normalize H to prevent division by zero
    eps = 1e-6
    if np.abs(H[2, 2]) < eps:
        H[2, 2] = eps
    H = H / H[2, 2]
    
    print("Homography Matrix:\n", H)
    return H

def warpImages(left, right, H):
    """
    Create a panorama by warping the right image into the left image's coordinate frame.
    The output canvas is sized to fit both images.
    """
    # Get image shapes
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    
    # Corners of the right image in homogeneous coordinates.
    corners_right = np.array([
        [0, 0, 1],
        [w_right, 0, 1],
        [w_right, h_right, 1],
        [0, h_right, 1]
    ]).T  # shape 3 x 4
    
    # Warp corners using H (maps right -> left)
    warped_corners = H @ corners_right
    # Normalize by the last coordinate
    warped_corners /= warped_corners[2, :]

    # Combine with left image corners.
    left_corners = np.array([
        [0, 0, 1],
        [w_left, 0, 1],
        [w_left, h_left, 1],
        [0, h_left, 1]
    ]).T
    all_corners = np.hstack((left_corners, warped_corners))
    
    x_min, x_max = int(np.floor(all_corners[0, :].min())), int(np.ceil(all_corners[0, :].max()))
    y_min, y_max = int(np.floor(all_corners[1, :].min())), int(np.ceil(all_corners[1, :].max()))

    # Compute translation to shift panorama into positive coordinates.
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float32)

    # Size of panorama
    pano_w, pano_h = x_max - x_min, y_max - y_min

    # Warp right image into panorama canvas.
    warped_right = cv2.warpPerspective(right, T @ H, (pano_w, pano_h))
    # Warp left image using only the translation T.
    warped_left = cv2.warpPerspective(left, T, (pano_w, pano_h))

    # Blend the two warped images.
    mask_left = (warped_left.sum(axis=2) > 0).astype(np.float32)
    mask_right = (warped_right.sum(axis=2) > 0).astype(np.float32)
    overlap = (mask_left * mask_right) > 0

    panorama = warped_left.copy()
    panorama[warped_right.sum(axis=2) > 0] = warped_right[warped_right.sum(axis=2) > 0]

    # Smooth blending
    for c in range(3):
        panorama[:, :, c][overlap] = ((warped_left[:, :, c].astype(np.float32) + 
                                      warped_right[:, :, c].astype(np.float32)) / 2).astype(np.uint8)
    
    return panorama

def main():
    left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)  # Ensure color
    right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)  # Ensure color

    if left_img is None or right_img is None:
        print("Error: Could not load one or both images. Check file paths.")
        return
    
    H = getHomography(left_img, right_img)

    # Warp images and create the panorama
    pano = warpImages(left_img, right_img, H)

    # Ensure the panorama is in color
    if len(pano.shape) == 2:  # If grayscale, convert to BGR
        pano = cv2.cvtColor(pano, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, pano)
    cv2.imshow("Panorama", pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Panorama saved at {output_path}")

if __name__ == "__main__":
    main()
