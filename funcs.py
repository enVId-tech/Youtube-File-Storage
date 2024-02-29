import binascii
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from functools import reduce
from PIL import Image


def toBin(BASE, file_path, mode):
    with open(file_path, mode) as file:
        # Read the file
        data = file.read()
        # Convert the file to hex
        hex_data = binascii.hexlify(data.encode())
        # Convert hex to binary and return as a list of digits
        return list(bin(int(hex_data, BASE))[2:])  # Start from index 2 to remove '0b' prefix

def toTxt(pixels, BASE, file_path, mode):
    with open(file_path, mode) as file:
        # Join binary digits into a string and convert to hexadecimal
        hex_data = hex(int(''.join(map(str, pixels)), 2))
        # Decode hexadecimal data to text using UTF-8 encoding
        text_data = binascii.unhexlify(hex_data[2:]).decode('utf-8')
        # Write the text data to the file
        file.write(text_data)


def toImg(pixels, file):
    if not isinstance(pixels, np.ndarray):
        raise ValueError('Input must be a valid numpy array')

    # Convert '1's to white and others to black
    img = np.where(pixels == '1', 255, 0)

    # Save the image to a file
    plt.imsave(file, img)

    # Display the image
    # plt.imshow(img)
    # plt.show()

    return img

def fillRemainingElements(pixels, frame_height, frame_width):
    # Calculate the number of elements to fill
    remaining_elements = frame_height * frame_width - pixels.size

    # Fill the remaining elements with empty space
    if remaining_elements > 0:
        pixels = np.append(pixels, ['1'] * 8)
        remaining_elements -= 8
        pixels = np.append(pixels, ['0'] * remaining_elements)

    return pixels


def readImgToBin(file):
    # Read the image to binary, using distributed computing
    img = Image.open(file)
    img = np.array(img)
    img_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img)

    mod = SourceModule("""
        __global__ void toBin(int *img, int *pixels, int width, int height) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
            int k = i * width + j;
            if (i < width && j < height) {
                pixels[k] = img[k] > 0 ? 1 : 0;
            }
        }
    """)

    func = mod.get_function("toBin")
    func(img_gpu,
         cuda.InOut(img),
         np.int32(img.shape[0]),
         np.int32(img.shape[1]),
         block=(16, 16, 1),
         grid=(16, 16, 1))
    cuda.memcpy_dtoh(img, img_gpu)
    img = img.reshape(
        (img.shape[0], img.shape[1], -1))  # Include color channels
    return img
