import os
import binascii
import torch
import numpy as np
from functools import reduce
from PIL import Image



def toTxt(pixels, BASE, file, mode):
    with open(file, mode) as file:
        # Convert the list of digits to a string of digits
        pixels = reduce(lambda x, y: str(x) + str(y), pixels)
        # Convert the string of digits to an integer
        pixels = int(pixels, 2)
        # Convert the integer to hex
        hex_data = hex(pixels)
        # Convert the hex to text
        text = binascii.unhexlify(hex_data[2:]).decode()
        # Write the text to a file
        file.write(text)

def readImgToBin(file):
    # Read the image to binary, using distributed computing
    if not os.path.exists(file):
        raise FileNotFoundError('File not found')

    if torch.cuda.is_available():
        return toBinGPU()
    else:
        return toBinCPU(file)

def toBinGPU(file):
    exit(0)

def toBinCPU(file):
    # Read the image to binary, using central processing
    img = Image.open(file)
    img = img.convert('1')
    img = np.array(img)
    img = img.flatten()
    return img