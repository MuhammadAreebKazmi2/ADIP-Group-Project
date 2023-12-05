import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading the input image
image = cv2.imread('punhal-khan-jamali-2019.jpg')

hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)

# Splitting the image into 3 channels; Hue, Saturation, and Intensity
h_channel, s_channel, i_channel = cv2.split(hsi_image)

# Applying Otsu thresholding to the H channel
h_level, h_binary = cv2.threshold(h_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_h = (h_binary)  # Inverting the binary image

# Applying Otsu thresholding to the S channel
s_level, s_binary = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_s = (s_binary)  # Inverting the binary image

# Applying Otsu thresholding to the I channel   
i_level, i_binary = cv2.threshold(i_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_i = (i_binary)  # Inverting the binary image

# Displaying the results for HSI
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(132)
plt.imshow(segmented_h, cmap='gray')
plt.title('Hue')

plt.subplot(133)
plt.imshow(segmented_s, cmap='gray')
plt.title('Saturation')

plt.show()

# Convert the image into HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Splitting the image into 3 channels; Hue, Saturation, and Value
h_channel, s_channel, v_channel = cv2.split(hsv_image)

# Applying Otsu thresholding to the H channel
h_level, h_binary = cv2.threshold(h_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_h = (h_binary)  # Inverting the binary image

# Applying Otsu thresholding to the S channel
s_level, s_binary = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_s = (s_binary)  # Inverting the binary image

# Applying Otsu thresholding to the V channel
v_level, v_binary = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_v = (v_binary)  # Inverting the binary image

# Displaying the results for HSV
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(segmented_h, cmap='gray')
plt.title('Hue')

plt.subplot(132)
plt.imshow(segmented_s, cmap='gray')
plt.title('Saturation')

plt.subplot(133)
plt.imshow(segmented_v, cmap='gray')
plt.title('Value')

plt.show()

# Convert the image into Lab color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Splitting the image into 3 channels; L, a, and b
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Applying Otsu thresholding to the L channel
l_level, l_binary = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_l = (l_binary)  # Inverting the binary image

# Applying Otsu thresholding to the a channel
a_level, a_binary = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_a = (a_binary)  # Inverting the binary image

# Applying Otsu thresholding to the b channel
b_level, b_binary = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
segmented_b = (b_binary)  # Inverting the binary image

# Displaying the results for Lab
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(segmented_l, cmap='gray')
plt.title('L')

plt.subplot(132)
plt.imshow(segmented_a, cmap='gray')
plt.title('a')

plt.subplot(133)
plt.imshow(segmented_b, cmap='gray')
plt.title('b')

plt.show()



# import cv2
# import numpy as np

# # Loading the input image
# image = cv2.imread('punhal-khan-jamali-2019.jpg')

# hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)

# # Splitting the image into 3 channels; Hue, Saturation, and Intensity
# h_channel, s_channel, i_channel = cv2.split(hsi_image)

# # Applying Otsu thresholding to the H channel
# h_level, h_binary = cv2.threshold(h_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_h = (h_binary)                # Inverting the binary image

# # Applying Otsu thresholding to the S channel
# s_level, s_binary = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_s = (s_binary)                # Inverting the binary image

# # Applying Otsu thresholding to the I channel   
# i_level, i_binary = cv2.threshold(i_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_i = (i_binary)                # Inverting the binary image

# # Displaying the results for HSI
# cv2.imshow('Original Image', image)
# cv2.imshow('Hue', segmented_h)    
# cv2.imshow('Saturation', segmented_s)
# cv2.imshow('Intensity', segmented_i)

# # Convert the image into HSV color space
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Splitting the image into 3 channels; Hue, Saturation, and Value
# h_channel, s_channel, v_channel = cv2.split(hsv_image)

# # Applying Otsu thresholding to the H channel
# h_level, h_binary = cv2.threshold(h_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_h = (h_binary)                # Inverting the binary image

# # Applying Otsu thresholding to the S channel
# s_level, s_binary = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_s = (s_binary)                # Inverting the binary image

# # Applying Otsu thresholding to the V channel
# v_level, v_binary = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_v = (v_binary)                # Inverting the binary image

# # Displaying the results for HSV
# cv2.imshow('Hue', segmented_h)
# cv2.imshow('Saturation', segmented_s)
# cv2.imshow('Value', segmented_v)

# # Convert the image into Lab color space
# lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# # Splitting the image into 3 channels; L, a, and b
# l_channel, a_channel, b_channel = cv2.split(lab_image)

# # Applying Otsu thresholding to the L channel
# l_level, l_binary = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_l = (l_binary)                # Inverting the binary image

# # Applying Otsu thresholding to the a channel
# a_level, a_binary = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_a = (a_binary)                # Inverting the binary image

# # Applying Otsu thresholding to the b channel
# b_level, b_binary = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# segmented_b = (b_binary)                # Inverting the binary image

# # Displaying the results for Lab
# cv2.imshow('L', segmented_l)
# cv2.imshow('a', segmented_a)
# cv2.imshow('b', segmented_b)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Good ones: L, Value, Intensity