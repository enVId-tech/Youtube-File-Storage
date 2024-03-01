import os
import binascii
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from concurrent.futures import ThreadPoolExecutor
# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
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

def toImg(pixels, file):
    if not isinstance(pixels, np.ndarray):
        raise ValueError('Input must be a valid numpy array')

    # Use multiple threads to convert the pixels to an image using distributed computing to convert binary to a black and white image
    if torch.cuda.is_available():
        img = toImgGPU(pixels)
    else:
        img = toImgCPU(pixels, num_threads=8)

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
    if not os.path.exists(file):
        raise FileNotFoundError('File not found')
    
    if torch.cuda.is_available():
        return toBinGPU()
    else:
        return toBinCPU(file)

def toBinGPU(file):
    # # Read the image to binary, using distributed computing
    # img = mpimg.imread(file)
    # img = np.array(img)
    # img_gpu = cuda.mem_alloc(img.nbytes)
    # cuda.memcpy_htod(img_gpu, img)

    # mod = SourceModule("""
    #     __global__ void toBin(int *img, int *pixels, int width, int height) {
    #         int i = threadIdx.x + blockIdx.x * blockDim.x;
    #         int j = threadIdx.y + blockIdx.y * blockDim.y;
    #         int k = i * width + j;
    #         if (i < width && j < height) {
    #             pixels[k] = img[k] > 0 ? 1 : 0;
    #         }
    #     }
    # """)

    # func = mod.get_function("toBin")
    # func(img_gpu,
    #      cuda.InOut(img),
    #      np.int32(img.shape[0]),
    #      np.int32(img.shape[1]),
    #      block=(16, 16, 1),
    #      grid=(16, 16, 1))
    # cuda.memcpy_dtoh(img, img_gpu)
    # img = img.reshape(
    #     (img.shape[0], img.shape[1], -1))  # Include color channels
    # return img
    exit(0)

def toBinCPU(file):
    # Read the image to binary, using central processing
    img = Image.open(file)
    img = img.convert('1')
    img = np.array(img)
    img = img.flatten()
    return img

def toImgGPU(pixels):
    # # Convert the pixels to an image, using distributed computing
    # img = np.array(pixels, dtype=np.uint8)
    # img_gpu = cuda.mem_alloc(img.nbytes)
    # cuda.memcpy_htod(img_gpu, img)

    # mod = SourceModule("""
    #     __global__ void toImg(int *pixels, int *img, int width, int height) {
    #         int i = threadIdx.x + blockIdx.x * blockDim.x;
    #         int j = threadIdx.y + blockIdx.y * blockDim.y;
    #         int k = i * width + j;
    #         if (i < width && j < height) {
    #             img[k] = pixels[k] > 0 ? 255 : 0;
    #         }
    #     }
    # """)

    # func = mod.get_function("toImg")
    # func(cuda.InOut(pixels),
    #      img_gpu,
    #      np.int32(pixels.shape[0]),
    #      np.int32(pixels.shape[1]),
    #      block=(16, 16, 1),
    #      grid=(16, 16, 1))
    # cuda.memcpy_dtoh(img, img_gpu)
    # img = img.reshape(
    #     (img.shape[0], img.shape[1], -1))  # Include color channels
    # return img
    exit(0)

def toImgCPU(pixels, num_threads=8):
    # Function to convert pixels to an image
    def convert_pixels(pixels):
        img = np.array(pixels, dtype=np.uint8)
        img = img * 255
        img = img.reshape((img.shape[0], img.shape[1], -1))  # Include color channels
        return img
    
    # Split the work into chunks
    chunk_size = len(pixels) // num_threads  # Adjust num_threads as needed
    chunks = [pixels[i:i+chunk_size] for i in range(0, len(pixels), chunk_size)]
    
    # Process each chunk concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        result_chunks = executor.map(convert_pixels, chunks)
    
    # Concatenate the results
    result = np.concatenate(list(result_chunks))
    return result
