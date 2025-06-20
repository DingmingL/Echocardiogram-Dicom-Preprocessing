import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import argparse


def convert_rgb2ybr(pixel_array: np.ndarray) -> np.ndarray:
    new_array = np.empty_like(pixel_array)
    for i in range(pixel_array.shape[0]):
        new_array[i] = cv2.cvtColor(pixel_array[i], cv2.COLOR_RGB2YCrCb)
    return new_array

def normalize(value: float, maximum: int, minimum: int) -> int:
    return int(255 * ((value - minimum) / (maximum - minimum)))\

def bounding_box(points: list) -> dict:
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)

    return {'min_x':bot_left_x, 'min_y':bot_left_y, 'max_x': top_right_x, 'max_y':top_right_y}

def create_mask(gray_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    shape_of_frames = gray_frames.shape
    changes = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    changes_frequency = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    binary_mask = np.zeros((shape_of_frames[1], shape_of_frames[2]))

    for i in range(len(gray_frames) - 1):
        diff = abs(gray_frames[i] - gray_frames[i + 1])
        changes += diff
        nonzero = np.nonzero(diff)
        changes_frequency[nonzero[0], nonzero[1]] += 1

    max_of_changes = np.amax(changes)
    min_of_changes = np.min(changes)

    for r in range(len(changes)):
        for p in range(len(changes[r])):
            if int(changes_frequency[r][p]) < 10:
                changes[r][p] = 0
            else:
                changes[r][p] = normalize(changes[r][p], max_of_changes, min_of_changes)

    nonzero_values_for_binary_mask = np.nonzero(changes)

    binary_mask[nonzero_values_for_binary_mask[0], nonzero_values_for_binary_mask[1]] += 1
    kernel = np.ones((5, 5), np.int32)
    erosion_on_binary_msk = cv2.erode(binary_mask, kernel, iterations=1)
    binary_mask_after_erosion = np.where(erosion_on_binary_msk, binary_mask, 0)

    nonzero_values_after_erosion = np.nonzero(binary_mask_after_erosion)
    binary_mask_coordinates = np.array([nonzero_values_after_erosion[0], nonzero_values_after_erosion[1]]).T
    binary_mask_coordinates = list(map(tuple, binary_mask_coordinates))
    bbox = bounding_box(binary_mask_coordinates)
    cropped_mask = binary_mask_after_erosion[int(bbox['min_x']):int(bbox['max_x']),
                    int(bbox['min_y']):int(bbox['max_y'])]

    for row in cropped_mask:
        ids = [i for i, x in enumerate(row) if x == 1]
        if len(ids) < 2:
            continue
        row[ids[0]:ids[-1]] = 1

    return cropped_mask, erosion_on_binary_msk, bbox

def save_video(dcm, fps, path_to_save):
    pixel_array = dcm.pixel_array
    if dcm.PhotometricInterpretation == 'RGB':
        pixel_array = convert_rgb2ybr(pixel_array)

    if pixel_array.shape[0] < 10:
        raise ValueError('Video is too short!')

    gray_frames = np.zeros(pixel_array.shape)
    gray_frames = pixel_array[:, :, :, 0]

    cropped_mask, erosion_on_binary_mask, bbox = create_mask(gray_frames)
    #print(bbox)

    height = int(bbox['max_x']) - int(bbox['min_x'])
    width = int(bbox['max_y']) - int(bbox['min_y'])

    # Define the codec and create VideoWriter object
    # 'XVID' is a popular codec for AVI files. You can also try 'MJPG'.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # The second parameter is the frames per second (fps). Adjust as needed.
    video = cv2.VideoWriter(path_to_save, fourcc, fps, (width, height))

    for gray_frame in gray_frames:
        masked_image = np.where(erosion_on_binary_mask, gray_frame, 0)
        cropped_image = masked_image[int(bbox['min_x']):int(bbox['max_x']),
                       int(bbox['min_y']):int(bbox['max_y'])]
        frame_color = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
        video.write(frame_color)

    # Release the VideoWriter and close any open windows
    video.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Generate masked AVI videos from DICOM files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing DICOM files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save generated AVI videos."
    )
    parser.add_argument(
        "--annotation_file_path",
        type=str,
        help="Annotation file path."
    )
    args = parser.parse_args()

    train_dir = args.input_dir
    train_m = args.output_dir
    annotation_file_path = args.annotation_file_path
    annotation = pd.read_csv(annotation_file_path)

    # Create output directory if it doesn't exist
    if not os.path.exists(train_m):
        os.makedirs(train_m)

    for file in os.listdir(train_dir):
        if file.lower().endswith(".dcm"):
            dcm = pydicom.dcmread(os.path.join(train_dir, file))
            filename = file.split('.')[0]
            output_path = os.path.join(train_m, filename + '.avi')
            if annotation_file_path:
                fps = annotation.FPS[annotation.FileName==filename].iloc[0]
                save_video(dcm, fps, path_to_save=output_path)
            else:
                save_video(dcm, 50, path_to_save=output_path)

if __name__ == '__main__':
    main()