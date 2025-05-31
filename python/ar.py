import numpy as np
import cv2
import os
from loadSaveVid import loadVid, saveVid
from matchPics import matchPicsCached
from planarH import computeH_ransac, compositeH

# Paths for inputs/outputs
book_mov_path = '../data/book.mov'
ar_source_path = '../data/ar_source.mov'
cv_cover_path = '../data/cv_cover.jpg'
output_video_path = '../results/ar.avi'
results_dir = '../results/'

# Global variables for caching and state
cachedLocs1 = None
cachedDesc1 = None
prevH = None

def cropFrameToCover(frame, cover):
    """
    Scale the AR frame so that it completely fills the cover dimensions,
    then center-crop it to exactly match the cover's size.
    """
    scale = max(cover.shape[1] / frame.shape[1], cover.shape[0] / frame.shape[0])
    new_w = int(frame.shape[1] * scale)
    new_h = int(frame.shape[0] * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    start_x = (new_w - cover.shape[1]) // 2
    start_y = (new_h - cover.shape[0]) // 2
    cropped = resized[start_y:start_y+cover.shape[0], start_x:start_x+cover.shape[1]]
    return cropped

def removeBlackBorders(frame, row_thresh=10):
    """
    Remove top and bottom black bars from the frame using contours.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, row_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        frame_cropped = frame[y:y+h, x:x+w]
        return frame_cropped
    else:
        return frame

def smoothHomography(H_curr, H_prev, alpha=0.9):
    """
    Blend the current homography with the previous one for stability.
    """
    if H_prev is None:
        return H_curr
    return alpha * H_prev + (1 - alpha) * H_curr

def main():
    os.makedirs(results_dir, exist_ok=True)

    # Load frames from cache if available.
    if os.path.exists("../data/arFrames.npy") and os.path.exists("../data/bookFrames.npy"):
        ar_frames = np.load("../data/arFrames.npy")
        book_frames = np.load("../data/bookFrames.npy")
    else:
        ar_frames = loadVid(ar_source_path)
        book_frames = loadVid(book_mov_path)
        np.save("../data/arFrames.npy", ar_frames)
        np.save("../data/bookFrames.npy", book_frames)

    cv_cover = cv2.imread(cv_cover_path)
    composite_frames = []
    global prevH, cachedLocs1, cachedDesc1

    for i in range(min(len(book_frames), len(ar_frames))):
        book_frame = book_frames[i]
        ar_frame = ar_frames[i]

        # 1) Remove black borders and crop the AR frame to match cv_cover.
        ar_frame_cleaned = removeBlackBorders(ar_frame)
        ar_frame_cropped = cropFrameToCover(ar_frame_cleaned, cv_cover)
        # Resize exactly to the cover's dimensions.
        ar_frame_cropped = cv2.resize(ar_frame_cropped, (cv_cover.shape[1], cv_cover.shape[0]))

        # 2) Use matchPicsCached to get consistent features for cv_cover.
        matches, locs1, locs2, cachedDesc1, _ = matchPicsCached(cv_cover, book_frame,
                                                                cachedLocs1=cachedLocs1,
                                                                cachedDesc1=cachedDesc1)
        cachedLocs1 = locs1

        # 3) Compute or reuse homography
        if len(matches) < 4 and prevH is not None:
            H = prevH
        else:
            x1 = locs1[matches[:, 0], :]
            x2 = locs2[matches[:, 1], :]
            # Convert from (row, col) to (x, y)
            x1_flip = x1[:, [1, 0]]
            x2_flip = x2[:, [1, 0]]
            H, inliers = computeH_ransac(x1_flip, x2_flip, nSamples=20, threshold=10)
            if H is None and prevH is not None:
                H = prevH
        if H is not None:
            H = H / H[2, 2]

        # 4) Smooth homography and composite
        H_smoothed = smoothHomography(H, prevH, alpha=0.9)
        composite_frame = compositeH(H_smoothed, ar_frame_cropped, book_frame)
        prevH = H_smoothed.copy()

        composite_frames.append(composite_frame)

        # 5) Display the composite frame in real-time
        cv2.imshow("Composite Frame", composite_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if i % 10 == 0:
            print(f"Processed frame {i}/{min(len(book_frames), len(ar_frames))}")

    cv2.destroyAllWindows()
    # 6) Save the final AR video
    composite_frames = np.array(composite_frames, dtype=np.uint8)
    np.save("../results/compositeFrames.npy", composite_frames)
    saveVid(output_video_path, composite_frames)
    print(f"AR video saved at {output_video_path}")

if __name__ == "__main__":
    main()
