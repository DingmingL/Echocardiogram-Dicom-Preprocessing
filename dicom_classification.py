import os
import numpy as np
import pandas as pd
import pydicom
import argparse
import shutil
import matplotlib.pyplot as plt
import logging
import sys
from datetime import datetime


def top10_pixel_means(region, channel_index, n=10):
    """
    For a given region (H x W x 3) and target channel (0 for red, 2 for blue),
    find the indices of the top n pixels based on the target channel,
    and return the mean [R, G, B] values of those pixels.
    """
    # Extract the target channel data from the region.
    channel_data = region[:, :, channel_index]
    flat = channel_data.flatten()
    num_pixels = flat.size
    if num_pixels < n:
        indices = np.arange(num_pixels)
    else:
        # Find indices of the top n values in the target channel.
        indices = np.argpartition(flat, -n)[-n:]
    # Reshape region to (H*W, 3) and get the corresponding pixels.
    region_reshaped = region.reshape(-1, 3)
    top_pixels = region_reshaped[indices]  # shape (n, 3)
    return np.mean(top_pixels, axis=0)


def is_grayscale_region(region, tol=10):
    """
    Determine if a given region is essentially grayscale: R ≈ G ≈ B.
    tol: tolerance for channel difference (absolute difference).
    Returns True if the region is grayscale, False otherwise.
    """
    # Extract each channel as integer arrays
    R = region[:, :, 0].astype(np.int32)
    G = region[:, :, 1].astype(np.int32)
    B = region[:, :, 2].astype(np.int32)

    # Compute absolute differences between channels
    diff_rg = np.abs(R - G)
    diff_rb = np.abs(R - B)
    diff_gb = np.abs(G - B)

    # print(f"Mean differences of Channels: {np.mean(diff_rg)}, {np.mean(diff_rb)}, {np.mean(diff_gb)}.")

    # If the average difference across all pixels is below tol for each pair,
    # consider the region grayscale.
    if (np.mean(diff_rg) < tol and
        np.mean(diff_rb) < tol and
        np.mean(diff_gb) < tol):
        return True
    return False

def classify_ultrasound_image(image, 
                              threshold_mid_red=200, 
                              threshold_mid_blue=200, 
                              dominance_margin=30, 
                              gray_tol=10):
    """
    Classify an RGB ultrasound image based on regional color distributions.

    Rules:
      1. Top-right region grayscale check:
         - If R ≈ G ≈ B (differences < gray_tol) in the top-right area, classify as "grey".
      2. Middle region Doppler check:
         a) Color Doppler:
            - Find the top 10 pixels by red channel in the middle region.
            - If mean(R) >= threshold_mid_red AND mean(G) < threshold_others AND mean(B) < threshold_others
              AND G ≈ B (difference < gray_tol), classify as "color_doppler".
         b) Continuous Doppler:
            - Find the top 10 pixels by blue channel in the middle region.
            - If mean(B) >= threshold_mid_blue AND mean(R) < threshold_others AND mean(G) < threshold_others
              AND R ≈ G (difference < gray_tol), classify as "continuous_doppler".
      3. If none of the above conditions are met, return "cannot_classify".

    Parameters:
      image (np.ndarray): RGB image of shape (H, W, 3)
      threshold_mid_red (float): Red channel threshold for Color Doppler in the middle region.
      threshold_mid_blue (float): Blue channel threshold for Continuous Doppler in the middle region.
      threshold_others (float): Maximum allowed mean for the non-dominant channels.
      gray_tol (float): Tolerance for channel difference used in grayscale check.

    Returns:
      str: "grey", "color_doppler", "continuous_doppler", or "cannot_classify".
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3D RGB numpy array.")
    
    H, W, _ = image.shape

    # --- 1. Grayscale check in top-right region ---
    # top_right = image[0:H//2, W//2:W, :]
    middle = image  # [H//2:5*H//6, W//4:3*W//4, :]
    if is_grayscale_region(middle, tol=gray_tol):
        logging.info("Middle region is grayscale.")
        return "grey"
    
    # --- 2. Doppler check in middle region ---
    # middle = image[H//2:5*H//6, W//4:3*W//4, :]

    # 2.1 Color Doppler check (using red channel)
    top_mid_red = top10_pixel_means(middle, channel_index=0)
    red_mean, green_mean_at_red, blue_mean_at_red = top_mid_red
    logging.info(f"Color Doppler candidates (R, G, B): {top_mid_red}")

    if (red_mean >= threshold_mid_red and
        red_mean - max(green_mean_at_red, blue_mean_at_red) >= dominance_margin):
        return "color_doppler"

    # 3. Continuous Doppler check (Blue dominant)
    top_mid_blue = top10_pixel_means(middle, channel_index=2)
    red_mean_at_blue, green_mean_at_blue, blue_mean = top_mid_blue
    logging.info(f"Continuous Doppler candidates (R, G, B): {top_mid_blue}")

    if (blue_mean >= threshold_mid_blue and
        blue_mean - min(red_mean_at_blue, green_mean_at_blue) >= dominance_margin):
        return "continuous_doppler"

    return "cannot_classify"


def dicom_classify(folder_path, file, test=False):
    """
    Classify DICOM files in folder_path into two subfolders based on the 
    (0028,0014) 'Ultrasound Color Data Present' tag.
    
    Subfolders created:
      - color_video: Files with Ultrasound Color Data Present (truthy value)
      - grey_video:  Files without or with a false value for the tag
    
    Parameters:
      folder_path (str): Path to the folder containing DICOM files.
    """
    # Define the target subfolder paths
    color_video_folder = os.path.join(folder_path, 'color_video')
    grey_video_folder = os.path.join(folder_path, 'grey_video')
    photo_folder = os.path.join(folder_path, 'photo')
    not_classified_folder = os.path.join(folder_path, 'cannot_classified')
    
    # Create the subfolders if they don't already exist
    os.makedirs(color_video_folder, exist_ok=True)
    os.makedirs(grey_video_folder, exist_ok=True)
    os.makedirs(photo_folder, exist_ok=True)
    os.makedirs(not_classified_folder, exist_ok=True)
    
    filename = file.split('/')[2] + '_' + file.split('/')[-1]
    file_path = os.path.join(folder_path, file)
    
    # Only process files (skip directories)
    if filename[-4:] == '.dcm':
            
        # Attempt to read the file as a DICOM dataset
        ds = pydicom.dcmread(file_path, force=True)
        # Retrieve the "Ultrasound Color Data Present" element
        element = ds.get((0x0028, 0x0014))
        file_type = ds.get((0x0008,0x0016))
        
        # Multi-frame DICOM image
        if file_type.value == "1.2.840.10008.5.1.4.1.1.3.1":
            # Determine the destination folder based on the element's value
            if element is not None and element.value:
                dest_folder = os.path.join(color_video_folder)
            else:
                dest_folder = os.path.join(grey_video_folder)
        
        # One frame DICOM image
        elif file_type.value == "1.2.840.10008.5.1.4.1.1.6.1":
            classified_label = classify_ultrasound_image(ds.pixel_array)
            dest_folder = os.path.join(photo_folder, classified_label)
            

        else:
            dest_folder = os.path.join(not_classified_folder)
            # os.makedirs(dest_folder, exist_ok=True)


        if test:
            #if "photo/cannot_classify" in dest_folder:
            print(f"Testing... Moved {filename} to {dest_folder}")
            
        else:
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(file_path, os.path.join(dest_folder, filename))
            logging.info(f"Moved {filename} to {dest_folder}")



def main():

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f"ultrasound_classification_{timestamp}.log"
    failed_path_file = f"failed_paths_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Starting DICOM classification task...")

    parser = argparse.ArgumentParser(
        description="Generate masked AVI videos from DICOM files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing DICOM files."
    )
    args = parser.parse_args()

    root_dir = args.input_dir

    dcm_paths = pd.read_csv(f"{root_dir}echo-record-list.csv")['dicom_filepath']

    failed_paths = []

    for rel_path in dcm_paths:
        full_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(full_path):
            logging.warning(f"File not found: {full_path}")
            failed_paths.append(full_path)
            continue

        try:
            dicom_classify(root_dir, rel_path, test=False)
        except Exception as e:
            logging.error(f"Error processing file {rel_path}: {e}", exc_info=True)
            failed_paths.append(full_path)

    if failed_paths:
        with open(failed_path_file, 'w') as f:
            for path in failed_paths:
                f.write(path + '\n')
        logging.info(f"Saved failed file paths to {failed_path_file}")
    else:
        logging.info("All files processed successfully.")

if __name__ == "__main__":
    main()
