import numpy as np
import cv2

"""
Q3.6
Compute the homography between two sets of points

@param[in] x1 Set of points x1 in heterogeneous coords (N×2)
@param[in] x2 Set of points x2 in heterogeneous coords (N×2)

@return H2to1 Closest 3x3 H matrix (least squares)
"""
def computeH(x1, x2):
    # Number of correspondences
    N = x1.shape[0]
    
    # Build matrix A from point correspondences:
    # For each correspondence (x1,y1) in image1 and (x2,y2) in image2,
    # we create two equations:
    #   [ x2, y2, 1, 0,  0,  0, -x1*x2, -x1*y2, -x1 ]
    #   [  0,  0, 0, x2, y2, 1, -y1*x2, -y1*y2, -y1 ]
    A = []
    for i in range(N):
        x1_i, y1_i = x1[i]
        x2_i, y2_i = x2[i]
        row1 = [x2_i, y2_i, 1, 0, 0, 0, -x1_i*x2_i, -x1_i*y2_i, -x1_i]
        row2 = [0, 0, 0, x2_i, y2_i, 1, -y1_i*x2_i, -y1_i*y2_i, -y1_i]
        A.append(row1)
        A.append(row2)
    A = np.array(A)
    
    # Solve A h = 0 using SVD. The solution is the singular vector corresponding 
    # to the smallest singular value.
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # 9-element vector
    
    # Reshape into 3x3 matrix and normalize so that H[2,2] == 1
    H2to1 = h.reshape((3, 3))
    H2to1 = H2to1 / H2to1[2, 2]
    
    return H2to1


"""
Q3.7
Normalize the coordinates to reduce noise before computing H

@param[in] x1 Set of points x1 in heterogeneous coords (N×2)
@param[in] x2 Set of points x2 in heterogeneous coords (N×2)

@return H2to1 Closest 3x3 H matrix (least squares)
"""
def computeH_norm(_x1, _x2):
    x1 = np.array(_x1)
    x2 = np.array(_x2)
    
    # Compute the centroid (mean) of the points
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)
    
    # Shift the origin of the points to the centroid
    x1_shifted = x1 - mean1
    x2_shifted = x2 - mean2
    
    # Compute the maximum Euclidean distance from the origin for each set
    dists1 = np.sqrt(np.sum(x1_shifted**2, axis=1))
    dists2 = np.sqrt(np.sum(x2_shifted**2, axis=1))
    max1 = np.max(dists1)
    max2 = np.max(dists2)
    
    # Compute scale factors to normalize so that the maximum distance equals sqrt(2)
    s1 = np.sqrt(2) / max1
    s2 = np.sqrt(2) / max2
    
    # Normalize the points
    x1norm = x1_shifted * s1
    x2norm = x2_shifted * s2
    
    # Construct the similarity transform matrices
    # For x1: mapping from original to normalized:
    T1 = np.array([[s1, 0, -s1 * mean1[0]],
                   [0, s1, -s1 * mean1[1]],
                   [0, 0, 1]])
    # For x2:
    T2 = np.array([[s2, 0, -s2 * mean2[0]],
                   [0, s2, -s2 * mean2[1]],
                   [0, 0, 1]])
    # Precomputed inverse of T1 for denormalization:
    T1_inv = np.linalg.inv(T1)
    
    # Compute homography on the normalized coordinates
    H_norm = computeH(x1norm, x2norm)
    
    # Denormalization: Map the homography back to the original coordinates
    H2to1 = T1_inv @ H_norm @ T2
    
    return H2to1


"""
Q3.8
Run RANSAC on set of matched points x1, x2.
Reduces effect of outliers by finding inliers.
Returns best fitting homography H and best inlier set.

@param[in] x1 Set of points x1 in heterogeneous coords (N×2)
@param[in] x2 Set of points x2 in heterogeneous coords (N×2)
@param[in] threshold: Squared error threshold (e.g., 10)
         Note: error is computed as the squared Euclidean distance.
@param[in] nSamples Number of iterations (e.g., 1000)

@return bestH2to1 Best homography matrix
@return bestInlier Vector of length N with a 1 at inlier matches, 0 elsewhere
"""
def computeH_ransac(_x1, _x2, nSamples=None, threshold=10):
    x1 = np.array(_x1)
    x2 = np.array(_x2)
    
    nPoints = len(x1)
    if nPoints < 4:
        print("Not enough points for homography estimation; returning identity matrix.")
        return np.eye(3), np.zeros(nPoints, dtype=np.uint8)
    
    if nSamples is None:
        nSamples = 1000
        
    bestInlierCount = 0
    bestH2to1 = None
    bestInliers = None
    
    # Run RANSAC iterations
    for i in range(nSamples):
        # Choose 4 random unique points (minimum required)
        indexes = np.random.choice(np.arange(nPoints), size=4, replace=False)
        x1_sample = x1[indexes]
        x2_sample = x2[indexes]
        
        # Compute candidate homography using normalized points
        try:
            H_candidate = computeH_norm(x1_sample, x2_sample)
        except np.linalg.LinAlgError:
            continue  # Skip iteration if computation fails
        
        # Convert all x2 points to homogeneous coordinates
        x2_h = np.hstack([x2, np.ones((nPoints, 1))])
        # Warp x2 points using candidate homography
        x2_transformed = (H_candidate @ x2_h.T).T  # shape: nPoints x 3
        # Normalize homogeneous coordinates
        x2_transformed = x2_transformed / x2_transformed[:, [2]]
        
        # Compute squared Euclidean error between x1 and transformed x2 points
        errors = np.sum((x1 - x2_transformed[:, :2])**2, axis=1)
        
        # Determine inliers based on the threshold
        inliers = errors < threshold
        inlierCount = np.sum(inliers)
        
        if inlierCount > bestInlierCount:
            bestInlierCount = inlierCount
            bestH2to1 = H_candidate
            bestInliers = inliers.astype(np.uint8)  # 1 for inlier, 0 for outlier
            
    return bestH2to1, bestInliers



"""
Q3.9
Create a composite image after warping the template image on top
of the image using the homography

Note that the homography we compute is from the image to the template;
x_template = H2to1 * x_photo
"""
def compositeH(H2to1, template, img, alreadyInverted=False):
    # For warping the template to the image, we need to invert H2to1
    if not alreadyInverted:
        H1to2 = np.linalg.inv(H2to1)
    else:
        H1to2 = H2to1

    # Create a mask of the same size as the template (single channel)
    mask = np.ones(template.shape[:2], dtype=np.uint8) * 255

    # Warp the mask to the coordinate frame of img using H1to2
    warped_mask = cv2.warpPerspective(mask, H1to2, (img.shape[1], img.shape[0]))
    # Convert mask to float and normalize to [0,1]
    warped_mask = warped_mask.astype(np.float32) / 255.0

    # Warp the template image to the coordinate frame of img using H1to2
    templateWarped = cv2.warpPerspective(template, H1to2, (img.shape[1], img.shape[0]))

    # Composite the warped template onto the image using the warped mask:
    # For each pixel, use the warped template where the mask is high, 
    # and the original image where the mask is low.
    # Here we assume a 3-channel image; expand the mask dimensions accordingly.
    warped_mask_color = np.dstack([warped_mask] * 3)
    composite_img = (warped_mask_color * templateWarped + (1 - warped_mask_color) * img).astype(np.uint8)

    return composite_img
