import binascii
import cv2
import numpy as np
from functools import reduce
from text_funcs import toTxt
from txt_to_img import textToImage
from img_to_txt import imageToText

# Global constants
BASE = 16
FRAME_HEIGHT = 1080
FRAME_WIDTH = 1920
DEVICE = 'cpu'


# Put these in PATH in System Environment Variables
# 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe'
# 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64'


def main():
    # Convert the text file to an image
    txtToImg = textToImage(BASE, FRAME_HEIGHT, FRAME_WIDTH, './input_files/text.txt', 'output_files/image.png', DEVICE)
    imgToTxt = imageToText('./output_files/image.png', DEVICE)

    if txtToImg and imgToTxt:
        print('Text file and image converted successfully')
    else:
        if txtToImg:
            print('Text file converted to image successfully')
        else:
            print('Text file conversion to image failed')

        if imgToTxt:
            print('Image converted to text successfully')
        else:
            print('Image conversion to text failed')
    exit(0)


if __name__ == "__main__":
    main()
