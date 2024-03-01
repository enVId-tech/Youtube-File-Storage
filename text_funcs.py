import os
import base64
import binascii
import cv2
import torch
import numpy as np
from functools import reduce
from PIL import Image


def toTxt(binary_img, file, device='cpu'):
    # Convert binary image to text

    # Check if the binary image is a numpy array
    if not isinstance(binary_img, np.ndarray):
        raise ValueError('Input must be a valid numpy array')

    if not os.path.exists(file):
        raise FileNotFoundError('File not found')

    # Determine the conversion function based on the device
    binary_img = readBinToTxt(binary_img, device=device)

    # Convert the binary image array to string
    text = ''.join(map(str, binary_img))

    # Write the text to a file
    with open(file, 'w') as f:
        f.write(text)

    return True


def readBinToTxt(file, device='cpu'):
    # Read the binary image to text, using distributed computing
    if device == 'gpu':
        return toTxtGPU(file)
    else:
        return toTxtCPU(file)

def toTxtGPU(file):
    # Read the binary image to text using cuda
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    # Read the image file using OpenCV
    img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read the image file: {file}")

    # Convert the image to binary (thresholding)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Flatten the binary image array
    binary_pixels = binary_img.flatten()

    # Convert the binary image array to hexadecimal
    hex_data = binary_pixels.tobytes()

    # Allocate GPU memory
    pixels_gpu = cuda.mem_alloc(len(hex_data))
    hex_data_gpu = cuda.mem_alloc(len(hex_data))

    # Copy data to GPU memory
    cuda.memcpy_htod(pixels_gpu, binary_pixels)
    cuda.memcpy_htod(hex_data_gpu, hex_data)

    # Define CUDA kernel
    mod = SourceModule("""
        __global__ void toTxtGPU(char *pixels, char *hex_data, int pixels_size) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < pixels_size) {
                pixels[idx] = hex_data[idx];
            }
        }
    """)
    toTxtGPU_kernel = mod.get_function("toTxtGPU")

    # Define block and grid sizes
    block_size = 1024
    grid_size = (len(hex_data) + block_size - 1) // block_size

    # Call the CUDA kernel
    toTxtGPU_kernel(pixels_gpu,
                    hex_data_gpu,
                    np.int32(len(hex_data)),
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1))

    # Copy the result back to host memory
    cuda.memcpy_dtoh(binary_pixels, pixels_gpu)

    # Convert the binary data back to image array format
    binary_img = np.reshape(binary_pixels, binary_img.shape)

    return binary_img

def toTxtCPU(file):
    # Read the image
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read the image file: {file}")
    
    # Convert the image to binary
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    return binary_img

def readImgToBin(file, device='cpu'):
    # Read the image to binary, using distributed computing
    if not os.path.exists(str(file)):
        raise FileNotFoundError('File not found')

    if device == 'gpu':
        return toBinGPU(str(file))
    else:
        return toBinCPU(str(file))

def toBinGPU(file):
    # Read the image to binary using cuda
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    # Read the image
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read the image file: {file}")

    # Convert the image to binary (thresholding)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Flatten the binary image array
    binary_pixels = binary_img.flatten()

    # Convert the binary image array to hexadecimal
    hex_data = binary_pixels.tobytes()

    # Allocate GPU memory
    pixels_gpu = cuda.mem_alloc(len(hex_data))
    hex_data_gpu = cuda.mem_alloc(len(hex_data))

    # Copy data to GPU memory
    cuda.memcpy_htod(pixels_gpu, binary_pixels)
    cuda.memcpy_htod(hex_data_gpu, hex_data)

    # Define CUDA kernel
    mod = SourceModule("""
        __global__ void toBinGPU(char *pixels, char *hex_data, int pixels_size) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < pixels_size) {
                pixels[idx] = hex_data[idx];
            }
        }
    """)
    toBinGPU_kernel = mod.get_function("toBinGPU")

    # Define block and grid sizes
    block_size = 1024
    grid_size = (len(hex_data) + block_size - 1) // block_size

    # Call the CUDA kernel
    toBinGPU_kernel(pixels_gpu,
                    hex_data_gpu,
                    np.int32(len(hex_data)),
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1))
    
    # Copy the result back to host memory
    cuda.memcpy_dtoh(binary_pixels, pixels_gpu)

    # Convert the binary data back to image array format
    binary_img = np.reshape(binary_pixels, binary_img.shape)

    return binary_img


def toBinCPU(file):
    # Read the image
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read the image file: {file}")

    # Convert the image to binary
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    return binary_img
