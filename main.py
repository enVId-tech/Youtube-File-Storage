import binascii
import cv2
import numpy as np
from functools import reduce
from funcs import toBin, toTxt

# Global constants
BASE = 16
FRAME_HEIGHT = 1080
FRAME_WIDTH = 1920

def main():
    frame_pixels = np.array([])

    pixels = toBin(BASE, 'text.txt', 'r')

    # Open the file to read a text file to hex
    print(len(pixels))

    # Open the file to write a hex file to text
    toTxt(pixels, BASE, 'w', 'hex.txt')

    frame_pixels = np.append(frame_pixels, pixels)

    # Calculate the number of elements to fill
    remaining_elements = FRAME_HEIGHT * FRAME_WIDTH - frame_pixels.size

    # Fill the remaining elements with empty space
    if remaining_elements > 0:
        frame_pixels = np.append(frame_pixels, ['0'] * remaining_elements)

    # Reshape the array to the frame dimensions
    frame_pixels = frame_pixels.reshape(FRAME_HEIGHT, FRAME_WIDTH)

    print(frame_pixels.size)

if __name__ == "__main__":
    main()
