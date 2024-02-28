import binascii
import numpy as np
from functools import reduce

def toBin(BASE, file, rwm):
    with open(file, rwm) as file:
        # Read the file
        data = file.read()
        # Convert the file to hex
        hex_data = binascii.hexlify(data.encode())

        pixels = list(bin(int(hex_data, BASE)))

        return pixels

def toTxt(pixels, BASE, rwm, file):
    with open(file, rwm) as file:
        binaryArr = reduce(lambda x, y: x + y, pixels)
        hex_data = hex(int(binaryArr, BASE))[4:]  # Convert binary to hexadecimal
        # Decode hexadecimal data to text using UTF-8 encoding
        text_data = binascii.unhexlify(hex_data).decode('utf-8')
        # Write the text data to the file
        file.write(text_data)
