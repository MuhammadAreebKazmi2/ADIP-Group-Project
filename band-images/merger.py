import rasterio
import numpy as np
import matplotlib.pyplot as plt

import os

# Directory path where the TIFF files are located
directory_path = r"D:\Habib University\Github\ADIP-Project\band-images\Banded"

# List all TIFF files in the directory
band_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".TIF")]

# Read the raster data
bands_data = []

for path in band_paths:
    try:
        band = rasterio.open(path).read(1)
        bands_data.append(band)
        print(f"Successfully read: {path}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Check if any data was successfully read
if not bands_data:
    print("No valid data was read. Please check your file paths and file accessibility.")
else:

    # Define the indices for false color composite
    # red = bands_data[5]  # NIR band
    # green = bands_data[4]  # Red band
    # blue = bands_data[3]  # Green band

    # red = bands_data[4]  # Red band
    # green = bands_data[3]  # Green band
    # blue = bands_data[2]  # Blue band

    red = bands_data[6]  # Red band
    green = bands_data[5]  # Green band
    blue = bands_data[4]  # Blue band


    # Stack the bands to create the composite image
    false_color_composite = np.dstack((red, green, blue))

    # Normalize the values for display
    false_color_composite = false_color_composite.astype(np.float64)
    false_color_composite = false_color_composite / np.max(false_color_composite)

    # Display the false color composite
    plt.imshow(false_color_composite)
    plt.title('False Color Composite Image (Landsat 8)')
    plt.axis('off')
    plt.show()
