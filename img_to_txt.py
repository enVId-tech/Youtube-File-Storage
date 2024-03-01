import cv2
import numpy as np
from text_funcs import toTxt

def imageToText(imagePath):
    try:
        # Convert the binary to text
        toTxt(imagePath, 'output_files/text.txt', 'gpu')

        return True
    except FileNotFoundError as e:
        print(e)
        return False
