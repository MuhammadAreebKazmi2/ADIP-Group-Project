import cv2
import numpy as np

def quantify_deforested_area(binary_image, pixel_size_meters, save_path=None):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the total area of deforested land
    total_area_pixels = sum(cv2.contourArea(contour) for contour in contours)
    total_area_sq_meters = total_area_pixels * pixel_size_meters**2
    total_area_sq_kms = total_area_sq_meters / 1e6

    # Save the result if a path is provided
    if save_path is not None:
        result_image = np.zeros_like(binary_image)
        cv2.drawContours(result_image, contours, -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(save_path, result_image)

    return total_area_sq_kms

def calculate_pixel_size(total_area_sq_kms, image_shape):
    # Calculate the pixel size in meters
    total_area_sq_meters = total_area_sq_kms * 1e6
    pixel_size_meters = np.sqrt(total_area_sq_meters / np.prod(image_shape))
    return pixel_size_meters

# Load your binary images and convert to grayscale
binary_image1 = cv2.imread('seg-hyd-2009.jpg', cv2.IMREAD_GRAYSCALE)
binary_image2 = cv2.imread('seg-hyd-2023.jpg', cv2.IMREAD_GRAYSCALE)

# Apply image segmentation logic to obtain binary images
# (Replace this with your actual segmentation logic)
_, binary_image1 = cv2.threshold(binary_image1, 128, 255, cv2.THRESH_BINARY)
_, binary_image2 = cv2.threshold(binary_image2, 128, 255, cv2.THRESH_BINARY)

# Replace total_area_sq_kms with your actual total area
total_area_sq_kms = 6.78

# Calculate pixel size for both images
pixel_size_meters_image1 = calculate_pixel_size(total_area_sq_kms, binary_image1.shape)
pixel_size_meters_image2 = calculate_pixel_size(total_area_sq_kms, binary_image2.shape)

# Quantify deforested area for each image
area_image1 = quantify_deforested_area(binary_image1, pixel_size_meters_image1, save_path='deforestation_result1.png')
area_image2 = quantify_deforested_area(binary_image2, pixel_size_meters_image2, save_path='deforestation_result2.png')

# Display and print results
# print(f"Pixel Size Image 1: {pixel_size_meters_image1:.2f} meters")
# print(f"Deforested Area in Image 1: {area_image1:.2f} sq. km")

# print(f"Pixel Size Image 2: {pixel_size_meters_image2:.2f} meters")
# print(f"Deforested Area in Image 2: {area_image2:.2f} sq. km")

# You can further analyze or compare the results as needed
# Calculate forested area
forested_area_image1 = total_area_sq_kms - area_image1
forested_area_image2 = total_area_sq_kms - area_image2

# Display and print results
print(f"Forested Area in Image 1: {forested_area_image1:.2f} sq. km")
print(f"Forested Area in Image 2: {forested_area_image2:.2f} sq. km")

# Calculate deforested area change
deforested_area_change = area_image2 - area_image1

# Display and print results
print(f"Deforested Area Change: {deforested_area_change:.2f} sq. km")