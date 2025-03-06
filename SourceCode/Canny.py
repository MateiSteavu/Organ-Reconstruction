import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt


def read_dicom_image(file_path):
    """
    Reads a DICOM file and returns the pixel array as a float64 numpy array.
    Applies any rescale slope/intercept if present.
    """
    ds = pydicom.dcmread(file_path)
    # Read the image as float64 to preserve precision
    image = ds.pixel_array.astype(np.float64)
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        image = image * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return image


def gaussian_blur(image, kernel_size=5, sigma=1.4):
    """
    Applies Gaussian blur using OpenCV.
    The image is assumed to be in its original 16-bit range (processed as float64).
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred


def sobel_gradients(image):
    """
    Computes the gradients using Sobel filters.
    Returns both the gradient magnitude and the gradient direction (in radians).
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    angle = np.arctan2(grad_y, grad_x)
    return magnitude, angle


def non_maximum_suppression(magnitude, angle):
    """
    Thins the edges by preserving only local maxima along the gradient direction.
    """
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float64)
    # Convert angles from radians to degrees (range 0 to 180)
    angle_deg = angle * 180.0 / np.pi
    angle_deg[angle_deg < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                # Initialize neighbor magnitudes
                q = 0.0
                r = 0.0
                # Angle 0°
                if (0 <= angle_deg[i, j] < 22.5) or (157.5 <= angle_deg[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Angle 45°
                elif 22.5 <= angle_deg[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Angle 90°
                elif 67.5 <= angle_deg[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Angle 135°
                elif 112.5 <= angle_deg[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    Z[i, j] = magnitude[i, j]
                else:
                    Z[i, j] = 0
            except IndexError:
                pass
    return Z


def double_thresholding(img, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    """
    Applies double thresholding on the non-max suppressed image.
    The thresholds are computed relative to the maximum gradient magnitude.
    Instead of normalizing the image, the algorithm works directly on the 16-bit data.
    """
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.float64)
    # For a 16-bit image, you might wish to set strong edges to the maximum
    # observed gradient magnitude, and weak edges to a fraction of that.
    strong_value = img.max()
    weak_value = strong_value / 3.0

    # Identify strong and weak edge pixels
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img < highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong_value
    res[weak_i, weak_j] = weak_value

    return res, weak_value, strong_value


def hysteresis(img, weak, strong):
    """
    Performs edge tracking by hysteresis. Weak pixels that are connected to strong
    pixels are promoted to strong.
    """
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def canny_edge_detection(image, lowThresholdRatio=0.02, highThresholdRatio=0.1,
                         kernel_size=5, sigma=1.4):
    """
    Runs the full Canny edge detection pipeline on a 16-bit image,
    preserving the original resolution throughout processing.
    """
    # 1. Smooth the image
    blurred = gaussian_blur(image, kernel_size, sigma)
    # 2. Compute the gradients
    grad_magnitude, grad_angle = sobel_gradients(blurred)
    # 3. Thin edges using non-maximum suppression
    non_max_img = non_maximum_suppression(grad_magnitude, grad_angle)
    # 4. Apply double thresholding using thresholds computed from the gradient magnitude
    threshold_img, weak, strong = double_thresholding(non_max_img,
                                                      lowThresholdRatio, highThresholdRatio)
    # 5. Perform edge tracking by hysteresis
    final_img = hysteresis(threshold_img, weak, strong)
    return final_img


# Example usage
if __name__ == '__main__':
    # Provide the path to your DICOM file
    file_path = 'C:/Reconstruire/DataSet_Sliced/ST000001/SE000001/IM000001.dcm'
    image = read_dicom_image(file_path)

    # Run the custom Canny edge detection pipeline on the 16-bit image.
    edges = canny_edge_detection(image)

    # For display purposes only: show the original and edge images.
    # (The processing itself is done without normalization.)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original 16-bit DICOM Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edges (16-bit)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()