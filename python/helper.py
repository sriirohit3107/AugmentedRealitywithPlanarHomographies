import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature

PATCHWIDTH = 9

def briefMatch(desc1, desc2, ratio=0.8):
	matches = skimage.feature.match_descriptors(desc1, desc2, 'hamming', cross_check=True, max_ratio=ratio)
	return matches


def plotMatches(im1, im2, matches, locs1, locs2, saveTo=None, showImg=False):
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    h1, w1 = im1_gray.shape
    h2, w2 = im2_gray.shape
    height = max(h1, h2)
    width = w1 + w2
    
    new_img = np.zeros((height, width), dtype=np.uint8)
    new_img[:h1, :w1] = im1_gray
    new_img[:h2, w1:] = im2_gray

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(new_img, cmap='gray')
    ax.axis('off')
    
    for match in matches:
        idx1, idx2 = match
        # row1, col1 => from locs1
        row1, col1 = locs1[idx1]
        # row2, col2 => from locs2
        row2, col2 = locs2[idx2]

        # Shift the second image's x-coord by w1
        col2 += w1
        
        # Plot the line: x = col, y = row
        ax.plot([col1, col2], [row1, row2], 'r-', linewidth=1)

    if saveTo:
        plt.savefig(saveTo)
    if showImg:
        plt.show()
    else:
        plt.close(fig)



def makeTestPattern(patchWidth, nbits):
	np.random.seed(0)
	compareX = patchWidth*patchWidth * np.random.random((nbits,1))
	compareX = np.floor(compareX).astype(int)
	np.random.seed(1)
	compareY = patchWidth*patchWidth * np.random.random((nbits,1))
	compareY = np.floor(compareY).astype(int)

	return (compareX, compareY)


def computePixel(img, idx1, idx2, width, center):
	halfWidth = width // 2
	col1 = idx1 % width - halfWidth
	row1 = idx1 // width - halfWidth
	col2 = idx2 % width - halfWidth
	row2 = idx2 // width - halfWidth
	return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0


def computeBrief(img, locs):

    #patchWidth = 9
    #nbits = 256
    #compareX, compareY = makeTestPattern(patchWidth,nbits)
    #m, n = img.shape
    #
    #halfWidth = patchWidth//2
    #
    #locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    #desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])
    #
    #return desc, locs

    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape

    halfWidth = patchWidth//2

    part1 = np.logical_and(halfWidth <= locs[:, 0], locs[:, 0] < m-halfWidth)
    part2 = np.logical_and(halfWidth <= locs[:, 1], locs[:, 1] < n-halfWidth)
    locs = locs[np.logical_and(part1, part2), :]

    zipped = np.column_stack((compareX, compareY))
    col1 = zipped[:, 0] % patchWidth - halfWidth
    row1 = zipped[:, 0] // patchWidth - halfWidth
    col2 = zipped[:, 1] % patchWidth - halfWidth
    row2 = zipped[:, 1] // patchWidth - halfWidth
    center0_row1 = np.add.outer(locs[:, 0], row1).astype(int)
    center1_col1 = np.add.outer(locs[:, 1], col1).astype(int)
    center0_row2 = np.add.outer(locs[:, 0], row2).astype(int)
    center1_col2 = np.add.outer(locs[:, 1], col2).astype(int)

    desc = np.zeros((locs.shape[0], zipped.shape[0]))
    desc[img[center0_row1, center1_col1] < img[center0_row2, center1_col2]] = 1

    return desc, locs


def corner_detection(im, sigma=0.15):
	# fast method
	result_img = skimage.feature.corner_fast(im, PATCHWIDTH, sigma)
	locs = skimage.feature.corner_peaks(result_img, min_distance=1)
	return locs

