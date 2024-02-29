import binascii
import cv2
import numpy as np
from functools import reduce
from funcs import toBin, toTxt, toImg, fillRemainingElements, readImgToBin

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
    # toTxt(pixels, BASE, 'hex.txt', 'w')

    frame_pixels = np.append(frame_pixels, pixels)

    # Fill the remaining elements with empty space
    frame_pixels = fillRemainingElements(frame_pixels, FRAME_HEIGHT,
                                         FRAME_WIDTH)
    # Reshape the array to the frame dimensions
    frame_pixels = frame_pixels.reshape(FRAME_HEIGHT, FRAME_WIDTH)

    # print(frame_pixels.size)

    # Save the image to a file
    img = toImg(frame_pixels, 'image.png')

    # Read the image to binary
    pixels = readImgToBin('image.png')

    # Open the file to write a binary file to text
    toTxt(pixels, BASE, 'output.txt', 'w')

    exit(0)


if __name__ == "__main__":
    main()
