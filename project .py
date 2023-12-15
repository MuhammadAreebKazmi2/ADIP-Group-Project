# import spectral
# import matplotlib.pyplot as plt

# image = spectral.open_image("highhighImage.tif")
# band_index = 0  # Replace with the desired band index
# band = image.read_band(band_index)

# plt.imshow(band, cmap='gray')
# plt.title(f'Band {band_index + 1}')
# plt.show()



import rasterio

# Open the raster file
with rasterio.open('mytiff.tif') as src:
    # Get the transform object
    transform = src.transform

    # Extract pixel size (spatial resolution)
    pixel_width = transform.a
    pixel_height = -transform.e  # Negative because the Y-axis is usually top to bottom

# Print the pixel size
print(f"Pixel size (width): {pixel_width}")
print(f"Pixel size (height): {pixel_height}")



# --------------- the code written below is written is an attempt to implement an alternative to the
# --------------- FLASH method to clear out any dicrepency due to the weather or other method


# import numpy as np
# from Py6S import *
# import rasterio
# import os

# print(os.environ['SIXS_PATH'])


# # Set the path to the 6S executable
# os.environ['SIXS_PATH'] = '/path/to/6S/executable'  # Replace with the actual path

# # Initialize the 6S radiative transfer model
# s = SixS()


# # Initialize the 6S radiative transfer model
# s = SixS()

# # Define the atmospheric parameters
# s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.Tropical)

# # Specify the path to your Landsat 7 image file
# landsat_image_path = 'mytiff.tif'  # Replace with the path to your Landsat 7 image

# with rasterio.open(landsat_image_path) as src:
#     # Read the radiance data and geospatial information from the image
#     radiance_data = src.read(1)
#     transform = src.transform
#     wavelength_band3 = 0.483  # Specify the wavelength for Band 3 (adjust for your data)
#     pixel_size_meters = transform.a  # Assuming square pixels
    
#     # Calculate the solar and view angles (for example, from metadata)
#     solar_z = 30  # Solar zenith angle in degrees
#     view_z = 0  # View zenith angle in degrees

#     # Create a new output file for corrected reflectance
#     profile = src.profile
#     profile.update(dtype=rasterio.float32, count=1)

#     # Atmospheric correction
#     reflectance_data = np.zeros_like(radiance_data, dtype=np.float32)

#     for row in range(radiance_data.shape[0]):
#         for col in range(radiance_data.shape[1]):
#             radiance_value = radiance_data[row, col]
            
#             # Set the wavelength
#             s.wavelength = Wavelength(wavelength_band3, wavelength_band3)

#             # Set the illumination and viewing geometry
#             s.geometry = Geometry.User()
#             s.geometry.solar_z = solar_z
#             s.geometry.view_z = view_z
#             s.geometry.view_az = 0  # Assuming a view azimuth angle of 0 degrees

#             # Run 6S to compute TOA reflectance
#             s.run()
            
#             # Retrieve the TOA reflectance
#             toa_reflectance = s.outputs.atmos_corrected_reflectance

#             # Store the result in the reflectance data array
#             reflectance_data[row, col] = toa_reflectance

#     # Save the corrected reflectance data to a new GeoTIFF file
#     with rasterio.open('landsat7_toa_reflectance.tif', 'w', **profile) as dst:
#         dst.write(reflectance_data, 1)



import rasterio
import numpy as np
import matplotlib.pyplot as plt
import time

# Define the path to your TIF file
# tif_file = "myImage.TIF"
# tif_file = "Shayan_Pics_2/Band_3.tif"
# tif_file2 = "Shayan_Pics_2/Band_4.tif"

tif_file = "ex2b3.TIF"
tif_file2 = "ex2b4.TIF"

start_cpu_time = time.process_time()

# Open the TIF file using rasterio
with rasterio.open(tif_file) as src:
    # Read the image as a NumPy array
    red_band = src.read(1)  # Use 1 as the band number


with rasterio.open(tif_file2) as src:
    # Read the image as a NumPy array
    nir_band = src.read(1)  # Use 1 as the band number


band_diff = red_band - nir_band
print(band_diff, " is the band difference")


height, width = red_band.shape

print(f"Image Width: {width} pixels")
print(f"Image Height: {height} pixels")


# Use np.any() to check for non-zero values
non_zero_indices = np.any(red_band != 0)
# first_non_zero_index = np.where(image_data != 0)[0]

if non_zero_indices:
    # Get the first non-zero index
    index_of_first_non_zero = np.argmax(red_band != 0)
    print(f"The first non-zero value is at index {index_of_first_non_zero}")
else:
    print("There are no non-zero values in the array.")

# Now, 'image_data' is a NumPy array containing the pixel values from the TIF image.
# You can access individual pixel values as needed.

print(nir_band)

# Calculate NDVI
ndvi = (nir_band - red_band) / (nir_band + red_band)

# Ensure NDVI values are between -1 and 1
ndvi = np.clip(ndvi, -1, 1)

# Define a threshold to distinguish high and low NDVI values
# You can adjust this threshold based on your specific data
threshold = 0.2  # Example threshold value

# Create a binary mask to highlight high and low NDVI areas
high_ndvi_mask = ndvi > threshold
low_ndvi_mask = ndvi < -threshold
    
# Create an image to highlight high NDVI areas (e.g., green)
high_ndvi_highlight = np.zeros_like(ndvi)
high_ndvi_highlight[high_ndvi_mask] = 1  # Set high NDVI areas to 1 (or any other value)

# Create an image to highlight low NDVI areas (e.g., red)
low_ndvi_highlight = np.zeros_like(ndvi)
low_ndvi_highlight[low_ndvi_mask] = 1  # Set low NDVI areas to 1 (or any other value)

# Combine high and low NDVI highlights
highlight_image = np.dstack((low_ndvi_highlight, high_ndvi_highlight, np.zeros_like(ndvi)))

# timing CPU time end
end_cpu_time = time.process_time()

# print CPU time
elapsed_cpu_time = end_cpu_time - start_cpu_time

print(f"CPU time: {elapsed_cpu_time} seconds")

# Save or display the highlight image
output_file = "highlight_image_check_1.png"
plt.imsave(output_file, highlight_image)

# Optionally, display the highlight image using matplotlib
plt.imshow(highlight_image)
plt.axis('off')  # Turn off axis labels
plt.show()



# You can loop through all the pixels to process or analyze the values as required.


