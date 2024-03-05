import os
import binascii
import cv2
import numpy as np


def video_convert_to_file(video_path, device='cpu'):
    try:
        binary_list = []

        # Read the video to binary
        if device == 'gpu':
            binary_list = gpu_read_video_to_binary(video_path)
        else:
            binary_list = cpu_read_image_to_binary(video_path)

        # Convert the 2D numpy array to a 1D string
        binary_list = ''.join(map(str, binary_list))

        # Shrink the binary list
        shrinked = find_sequence(binary_list)
        title = find_title(binary_list)

        print(f"The title is: {title}")

        if not shrinked or not title:
            return False

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

    # Flatten the 2D array to a 1D array and convert it to a string
    binary_str = ''.join(map(str, binary_img.flatten()))

    return binary_str


def gpu_read_video_to_binary(video_path):
    try:
        # Read the video using gpu
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Could not read the video file: {video_path}")

        # Read the video to binary
        binary_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale image to binary (thresholding)
            _, binary_img = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY)

            # Flatten the 2D array to a 1D array and convert it to a string
            binary_str = ''.join(map(str, binary_img.flatten()))

            # Append the binary string to the list
            binary_list.append(binary_str)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        return binary_list
    except Exception as e:
        print(e)
        return False


def find_sequence(binary_list):
    try:
        if not isinstance(binary_list, str):
            raise TypeError('The binary list must be a string')

        # Define the sequence to find
        endOfBin = "000000010"

        # Convert endOfBin to a string
        endOfBin_str = ''.join(map(str, endOfBin))
        print(f"Finding {endOfBin} in {binary_list}")

        # Find the sequence
        sequence = binary_list.find(endOfBin_str)
        if sequence == -1:
            raise ValueError('The sequence was not found in the binary list')

        print(f"The length of the binary list is: {len(binary_list[:sequence])}")

        # Return the shrinked binary list
        return binary_list[sequence:]
    except Exception as e:
        print(e)
        return False


def find_title(binary_list):
    try:
        if not isinstance(binary_list, str):
            raise TypeError('The binary list must be a string')

        # Define the sequence to find
        endOfBin = "000000100"

        # Find the sequence
        sequence = binary_list.find(endOfBin)
        if sequence == -1:
            raise ValueError('The sequence was not found in the binary list')

        # Get the binary title
        binary_title = binary_list[:sequence]
        print(f"The binary title is: {binary_title}")
        
        exit(0)
        # Convert the binary title to a text string
        title = ''.join(
            chr(int(binary_title[i:i + 8], 2))
            for i in range(0, len(binary_title), 8))

        return title
    except Exception as e:
        print(e)
        return False
