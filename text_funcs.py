import os
import binascii
import cv2
import numpy as np
import concurrent.futures

def toTxt(imgFile, txtFile, device='cpu'):
    # Convert binary image to text

    if not os.path.exists(txtFile):
        os.makedirs(os.path.dirname(txtFile), exist_ok=True)

    # Determine the conversion function based on the device
    binaryList = readBinToTxt(imgFile, device=device)

    shrinked = findSequence(binaryList)

    # Convert the binary list to text
    text = toText(shrinked)

    # Write the text to a file
    with open(txtFile, 'w') as f:
        f.write(text)

    return True


def readBinToTxt(imgFile, device='cpu'):
    # Read the binary image to text, using distributed computing
    if device == 'gpu':
        return toTxtGPU(imgFile)
    else:
        return toTxtCPU(imgFile)
        # return toTxtIntegratedGraphics(imgFile, 'output_files/text.txt')


def toTxtGPU(file):
    # Read the image to binary using cuda
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    # Read the image
    img = cv2.imread(file)
    if img is None:
        raise FileNotFoundError(f"Could not read the image file: {file}")

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary (thresholding)
    _, binary_img = cv2.threshold(gray_img, 128, 1, cv2.THRESH_BINARY)

    # Flatten the binary image array
    binary_pixels = binary_img.flatten()

    # Allocate GPU memory
    pixels_gpu = cuda.mem_alloc(binary_pixels.nbytes)

    # Copy data to GPU memory
    cuda.memcpy_htod(pixels_gpu, binary_pixels)

    # Define CUDA kernel
    mod = SourceModule("""
        __global__ void toBinGPU(int *pixels, int pixels_size) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < pixels_size) {
                pixels[idx] = pixels[idx];
            }
        }
    """)
    toBinGPU_kernel = mod.get_function("toBinGPU")

    # Define block and grid sizes
    block_size = 1024
    grid_size = (binary_pixels.size + block_size - 1) // block_size

    # Call the CUDA kernel
    toBinGPU_kernel(pixels_gpu,
                    np.int32(binary_pixels.size),
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1))

    # Copy the result back to host memory
    cuda.memcpy_dtoh(binary_pixels, pixels_gpu)

    # Convert the binary data back to image array format
    binary_img = np.reshape(binary_pixels, binary_img.shape)

    # Convert the binary image to text
    binary_text = ''.join(str(pixel) for pixel in binary_pixels)

    return binary_text

def toTxtCPU(file, num_threads=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Read the image
        img = cv2.imread(file)
        if img is None:
            raise FileNotFoundError(f"Could not read the image file: {file}")

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to binary (thresholding)
        _, binary_img = cv2.threshold(gray_img, 128, 1, cv2.THRESH_BINARY)

        # Flatten the binary image array
        binary_pixels = binary_img.flatten()

        # Convert the binary image to text
        binary_text = ''.join(str(pixel) for pixel in binary_pixels)

        return binary_text

def findSequence(binary_list):
    # Convert the list to a string
    if type(binary_list) is not str:
        print('Binary list is not a string')
        exit(1)

    # Define the sequence to find
    sequence = '1111111101'

    # Find the index of the sequence in the string
    index = binary_list.find(sequence)

    # If the sequence was found, return the part of the string up to the sequence
    if index != -1:
        return binary_list[:index]
    else:
        print('Sequence not found')
        exit(1)


def toText(binary_list):
    # Join the binary list into a single string
    binary_str = ''.join(binary_list)

    # Convert the binary string to an integer
    int_data = int(binary_str, 2)

    # Convert the integer to hexadecimal
    hex_data = hex(int_data)[2:]  # Start from index 2 to remove '0x' prefix

    # Decode the hexadecimal to get the original text
    text = binascii.unhexlify(hex_data).decode()

    return text

def toTxtIntegratedGraphics(imgFile, txtFile):
    # Convert binary image to text using integrated graphics

    if not os.path.exists(txtFile):
        os.makedirs(os.path.dirname(txtFile), exist_ok=True)

    # Read the image
    img = cv2.imread(imgFile)
    if img is None:
        raise FileNotFoundError(f"Could not read the image file: {imgFile}")

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary (thresholding)
    _, binary_img = cv2.threshold(gray_img, 128, 1, cv2.THRESH_BINARY)

    # Flatten the binary image array
    binary_pixels = binary_img.flatten()

    # Convert the binary image to text
    text = ''.join(str(pixel) for pixel in binary_pixels)

    # Write the text to a file
    with open(txtFile, 'w') as f:
        f.write(text)

    return True