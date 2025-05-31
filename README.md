# Augmented Reality with Planar Homographies 🎯

Built a real-time AR system by estimating planar homographies from feature matches across 500+ frames. Implemented FAST + BRIEF for keypoint detection and binary descriptor matching, achieving >90% match accuracy. Applied RANSAC to filter outliers and computed essential matrices for 3D reconstruction. Successfully overlaid video content onto tracked objects and created panoramas with <2 px reprojection error.

## 📌 Features

- FAST keypoint detection and BRIEF descriptor extraction  
- Feature matching using Hamming distance  
- Homography estimation via RANSAC  
- AR overlay on real-world objects in images and video  
- Fundamental and essential matrix computation  
- 3D reconstruction from stereo correspondences  
- Panorama stitching (Extra Credit)

## 🛠️ Tools & Libraries

- Python  
- OpenCV  
- NumPy  
- Matplotlib (for visualization)

## 📁 Project Structure

├── matchPics.py # FAST + BRIEF feature matching pipeline
├── planarH.py # Homography estimation using RANSAC
├── HarryPotterize.py # Automated AR overlay on static images
├── ar.py # AR video integration
├── panorama.py # Panorama stitching (extra credit)



## 📷 Sample Output

- AR overlay on a book cover (static image)
- Live AR content mapped to moving book in video
- 3D point cloud reconstruction from epipolar geometry
- Stitched panorama from two images

## 📈 Performance Highlights

- **>90% feature match accuracy** with FAST + BRIEF  
- **Real-time AR overlays** across **500+ frames**  
- **<2 px reprojection error** in 3D reconstruction  
- Seamless panorama stitching using homographies

## 🚀 Getting Started

1. Clone the repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run a demo : 
python HarryPotterize.py
python ar.py

## 🧠 Concepts Covered

- Feature Detection & Description  
- Image Warping & Homography  
- Epipolar Geometry & 3D Reconstruction  
- AR Pipeline Implementation  
- Robust Matching (RANSAC)

## 📚 References

- Edward Rosten et al. (2010). *Faster and Better: A Machine Learning Approach to Corner Detection*  
- Michael Calonder et al. (2010). *BRIEF: Binary Robust Independent Elementary Features*  
- David Lowe (2004). *SIFT Features from Scale-Invariant Keypoints*
