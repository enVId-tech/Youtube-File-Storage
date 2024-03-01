import cv2
import numpy as np
from text_funcs import toTxt, readImgToBin

def imageToText(imagePath):
    try:
        # Read the image to binary
        pixels = readImgToBin(str(imagePath), 'gpu')

        # Convert the binary to text
        toTxt(pixels, 'output_files/text.txt', 'gpu')

        return True
    except FileNotFoundError as e:
        print(e)
        return False