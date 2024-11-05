import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import os


def detect_ridges(gray, sigma=3.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order="rc")
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


def plot_images(*images, output_dir=None, base_filename="output"):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True)
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap="gray")
        ax[i].axis("off")
        extent = ax[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(f"{output_dir}/{base_filename}_fig{i}.png", bbox_inches=extent)
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.close(fig)


def process_image(filepath, output_sub_dir):
    # Step 1: Import the image
    img = cv2.imread(filepath, 1)

    # Step 2: Sharpen the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)

    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Step 4: Perform histogram equalization
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    img2 = cdf[gray]

    # Step 5: Ridge detection
    a, b = detect_ridges(img2, sigma=2.7)
    plot_images(
        a,
        b,
        output_dir=output_sub_dir,
        base_filename=os.path.basename(filepath).split(".")[0],
    )

    # Step 6: Convert the ridge detection result to binary
    img = cv2.imread(
        f"{output_sub_dir}/{os.path.basename(filepath).split('.')[0]}_fig1.png", 0
    )
    bg = cv2.dilate(img, np.ones((5, 5), dtype=np.uint8))
    bg = cv2.GaussianBlur(bg, (5, 5), 1)
    src_no_bg = 255 - cv2.absdiff(img, bg)
    _, thresh = cv2.threshold(src_no_bg, 240, 255, cv2.THRESH_BINARY)

    # Step 7: Thinning / Skeletonizing
    thinned = cv2.ximgproc.thinning(thresh)
    cv2.imwrite(
        f"{output_sub_dir}/{os.path.basename(filepath).split('.')[0]}_thinned.png",
        thinned,
    )
    print(
        f"Processed and saved: {output_sub_dir}/{os.path.basename(filepath).split('.')[0]}_thinned.png"
    )


def main():
    input_dir = "./remove_bg/"
    output_dir = "./output/"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each subdirectory in the input directory
    for sub_dir in os.listdir(input_dir):
        sub_dir_path = os.path.join(input_dir, sub_dir)

        # Check if the item is a directory
        if os.path.isdir(sub_dir_path):
            # Create a corresponding directory in the output folder
            output_sub_dir = os.path.join(output_dir, sub_dir)
            os.makedirs(output_sub_dir, exist_ok=True)

            # Process each .tif file in the subdirectory
            for filename in os.listdir(sub_dir_path):
                if filename.endswith(".tif"):
                    filepath = os.path.join(sub_dir_path, filename)
                    process_image(filepath, output_sub_dir)


if __name__ == "__main__":
    main()
