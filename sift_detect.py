import numpy as np
import cv2 as cv
import glob
import os

# Minimum match count to consider a valid match
MIN_MATCH_COUNT = 15

# Path to the directory with thinned images
output_dir = "./output/"

# Load the input image to find feature points
input_img = cv.imread("./output/3/012_3_1_thinned.png")
input_img = input_img.astype("uint8")
gray = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(input_img, None)

flag = 0

# Loop through each subdirectory in the output folder
for sub_dir in os.listdir(output_dir):
    sub_dir_path = os.path.join(output_dir, sub_dir)

    # Check if the item is a directory
    if os.path.isdir(sub_dir_path):
        # Process each thinned .png file in the subdirectory
        for file in glob.glob(os.path.join(sub_dir_path, "*_thinned.png")):
            frame = cv.imread(file)
            frame = frame.astype("uint8")
            gray1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Detect keypoints and descriptors for the frame
            kp2, des2 = sift.detectAndCompute(frame, None)

            # Use FLANN based matcher
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(
                np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2
            )

            # Filter good matches using the ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # Calculate matching score as a decimal ratio of good matches to total keypoints in image
            score = len(good) / len(kp1) if kp1 else 0  # Ensure no division by zero
            if len(good) > MIN_MATCH_COUNT:
                print(f"Match {file}, score: {score}")
                flag = 1
            else:
                print(f"Not match {file}, score: {score}")

            # Draw matches
            matchesMask = [1] * len(good)  # Only draw "good" matches
            draw_params = dict(
                matchColor=(0, 255, 0),  # draw matches in green color
                singlePointColor=None,
                matchesMask=matchesMask,  # draw only good matches
                flags=2,
            )
            img3 = cv.drawMatches(input_img, kp1, frame, kp2, good, None, **draw_params)
            cv.imshow("Result", img3)
            cv.waitKey(0)
            cv.destroyAllWindows()

# Check if any matches were found
if flag == 0:
    print("No strong matches among the given set.")
