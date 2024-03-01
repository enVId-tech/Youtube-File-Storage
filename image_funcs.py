import binascii
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

def toBin(BASE, file_path, mode):
    with open(file_path, mode) as file:
        # Read the file
        data = file.read()
        # Convert the file to hex
        hex_data = binascii.hexlify(data.encode())
        # Convert hex to binary and return as a list of digits
        return list(bin(int(
            hex_data, BASE))[2:])  # Start from index 2 to remove '0b' prefix

def fillRemainingElements(pixels, frame_height, frame_width):
    # Calculate the number of elements to fill
    remaining_elements = frame_height * frame_width - pixels.size

    # Fill the remaining elements with empty space
    if remaining_elements > 0:
        pixels = np.append(pixels, ['1'] * 8)
        pixels = np.append(pixels, ['0'] * 1)
        pixels = np.append(pixels, ['1'] * 1)
        remaining_elements -= 10
        pixels = np.append(pixels, ['0'] * remaining_elements)

    return pixels


def toImg(pixels, file, height, width, device='cpu'):
    if not isinstance(pixels, np.ndarray):
        raise ValueError('Input must be a valid numpy array')

    # Use multiple threads to convert the pixels to an image using distributed computing to convert binary to a black and white image
    # img = toImgGPU(pixels)
    if device == 'gpu':
        img = toImgGPU(pixels)
    else:
        img = toImgCPU(pixels)

    # Ensure image data is in uint8 format
    img = img.astype(np.uint8)

    # Reshape the array
    img = img.reshape(
        (height, width))  # replace 'height' and 'width' with the actual values

    # Check if the file path is valid
    if not file.endswith('.png'):
        raise ValueError('File must have a .png extension')

    # Save the image to a file
    plt.imsave(file, img)

    return img


def toImgCPU(pixels, num_threads=8):
    # Convert the pixels to a numpy array
    pixels = np.array(pixels, dtype=np.int32)

    # Use multiple threads to convert the pixels to an image
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        img = list(executor.map(lambda x: x * 255, pixels))

    # Return the result
    return np.array(img, dtype=np.uint8)


def toImgGPU(pixels):
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    # Define the CUDA kernel
    mod = SourceModule("""
    __global__ void toImgGPU(int *pixels, int *result, int num_pixels) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < num_pixels) {
            result[i] = pixels[i] * 255;
        }
    }
    """)

    # Get the kernel function
    toImgGPU_kernel = mod.get_function("toImgGPU")

    # Convert the pixels to a numpy array
    pixels = np.array(pixels, dtype=np.int32)
    result = np.zeros_like(pixels)

    # Allocate memory on the device
    pixels_gpu = cuda.to_device(pixels)
    result_gpu = cuda.to_device(result)

    # Define block and grid sizes
    block_size = 1024
    grid_size = (pixels.size + block_size - 1) // block_size

    # Call the kernel
    toImgGPU_kernel(pixels_gpu,
                    result_gpu,
                    np.int32(pixels.size),
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1))

    # Copy the result back to the host
    cuda.memcpy_dtoh(result, result_gpu)

    # Return the result
    return result
