import os
import binascii
import cv2
import numpy as np


def video_convert_to_file(video_path, device='cpu'):
    try:
        binary_list = []

        # Determine the conversion function based on the device
        if device == 'gpu':
            binary_list = gpu_read_video_to_binary(video_path)
        else:
            binary_list = cpu_read_image_to_binary(video_path)

        # Convert the 2D numpy array to a 1D string
        binary_list = ''.join(binary_list.flatten().astype(str))

        # Shrink the binary list
        shrinked = find_sequence(binary_list)
        title = find_title(binary_list)

        if not shrinked or not title:
            return False

        title = title.decode('utf-8')

        # Write the binary list to a file
        with open(title, 'wb') as file:
            file.write(shrinked)

        return True
    except Exception as e:
        print(e)
        return False


def cpu_read_image_to_binary(image_path):
    # Read the image to binary using cpu
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read the image file: {image_path}")

    # Convert the grayscale image to binary (thresholding)
    _, binary_img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)

    return binary_img


def gpu_read_video_to_binary(image_path):
    import cv2
    import cupy as cp

    # Read the video
    video = cv2.VideoCapture(image_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Could not read the video file: {image_path}")

    # Read the video to binary
    binary_list = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_gpu = cp.asarray(frame)  # Move the frame to GPU
        gray_gpu = cp.dot(frame_gpu[..., :3],
                          [0.299, 0.587, 0.114])  # Convert to grayscale on GPU
        _, binary_img_gpu = cp.cuda.cutensor.threshold(
            gray_gpu, 128, 1,
            cp.cuda.cutensor.THRESH_BINARY)  # Apply threshold on GPU
        binary_list.append(binary_img_gpu.get(
        ))  # Move the binary image back to CPU and append to the list
    video.release()

    return binary_list


def find_sequence(binary_list):
    try:
        if not isinstance(binary_list, str):
            raise TypeError('The binary list must be a string')

        # Define the sequence to find
        endOfBin = bytes([0, 0, 0, 0, 0, 0, 0, 1, 0])

        # Find the sequence
        sequence = binary_list.find(endOfBin)
        if sequence == -1:
            raise ValueError('The sequence was not found in the binary list')

        # Return the sequence
        return binary_list[:sequence]
    except Exception as e:
        print(e)
        return False

def find_title(binary_list):
    try:
        if not isinstance(binary_list, str):
            raise TypeError('The binary list must be a string')

        # Define the sequence to find
        endOfBin = bytes([0, 0, 0, 0, 0, 0, 1, 0, 0])

        # Find the sequence
        sequence = binary_list.find(endOfBin)
        if sequence == -1:
            raise ValueError('The sequence was not found in the binary list')

        # Return the title
        return binary_list[:sequence]
    except Exception as e:
        print(e)
        return False
