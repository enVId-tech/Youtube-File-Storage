import sys
import numpy as np
import queue
import threading
import time
import psutil
from PIL import Image
import cv2

# Define the maximum size of each binary fragment
MAX_BINARY_SIZE = 1000000  # 1 MB
MAX_MEMORY = 2 * 10**9  # Memory limit in bytes
MAX_THREADS = 8  # Maximum number of threads to use


def file_convert_to_video(frame, input_file, output_file, frame_rate, device):
    try:
        height = frame[0]
        width = frame[1]

        sys.set_int_max_str_digits(100000000)
        binary_fragments = file_to_binary_bits(input_file)

        if device == 'gpu':
            img = gpu_binary_to_image(binary_fragments)
        else:
            img = cpu_binary_to_image(binary_fragments, MAX_MEMORY,
                                      MAX_THREADS)

        print(f'Frame Sample: {img[0:100]}')

        frames = handle_multiple_frames(img, height, width)

        print(f'Number of frames: {len(frames)}')

        # Fill the remaining elements for each frame
        for i in range(len(frames)):
            frames[i] = fill_remaining_elements(frames[i], height, width)

        # Save the frames as video
        video = create_video_from_frames(frames, output_file, height, width,
                                         frame_rate)

        if video:
            return True
        else:
            return False
    except FileNotFoundError as e:
        print(e)
        return False
    except Exception as e:
        print(e)
        return False


def create_video_from_frames(frames,
                             output_file,
                             frame_height,
                             frame_width,
                             frame_rate=30):
    try:
        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
        video_writer = cv2.VideoWriter(output_file,
                                       fourcc,
                                       frame_rate, (frame_width, frame_height),
                                       isColor=False)

        # Write frames to video
        for i in range(len(frames)):
            video_writer.write(frames[i].reshape(frame_height, frame_width))

        # Release resources
        video_writer.release()
        cv2.destroyAllWindows()

        print("Video saved successfully!")
        return True
    except Exception as e:
        print(e)
        return False


def handle_multiple_frames(pixel_array, frame_height, frame_width):
    # Convert the pixel array to a list of frames
    return [
        pixel_array[i:i + frame_height * frame_width]
        for i in range(0, len(pixel_array), frame_height * frame_width)
    ]


def file_to_binary_bits(file_path, chunk_size=8192):
    binary_bits = ""
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            binary_bits += ''.join(
                format(x, '08b') for x in bytearray(chunk))
    return binary_bits

def convert_binary_to_grayscale(img, start_idx, end_idx):
    for i in range(start_idx, end_idx):
        img[i] = img[i] * 255


def cpu_binary_to_image(fragments, num_threads=8):
    try:
        binary_string = ''.join(
            map(lambda x: bin(int.from_bytes(x, byteorder='big'))[2:],
                fragments))
        # Assuming 'binary_array' is your binary array
        binary_array = np.array(list(map(int, binary_string)), dtype=np.uint8)

        # Convert binary array to uint8
        image_array = binary_array.astype(np.uint8)

        # Convert binary array to grayscale
        for i in range(0, len(image_array), MAX_BINARY_SIZE):
            convert_binary_to_grayscale(image_array, i, i + MAX_BINARY_SIZE)

        return image_array
    except Exception as e:
        print(e)
        return None


# MUST BE RUN ON A GPU-ENABLED MACHINE
# This function uses CUDA to convert the binary array to an image
# Note: This function MUST get a binary array as input
def gpu_binary_to_image(binary_fragments):
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    # Concatenate binary fragments
    binary_data = b''.join(binary_fragments)

    # Create a PyCuda context
    context = cuda.Device(0).make_context()

    # Convert the binary data to a NumPy array
    binary_array = np.frombuffer(binary_data, dtype=np.uint8)

    # Allocate device memory for input data
    input_data = cuda.mem_alloc(binary_array.nbytes)
    cuda.memcpy_htod(input_data, binary_array)

    # Allocate device memory for output array
    output_size = len(binary_array)
    output_data = cuda.mem_alloc(output_size)

    # Compile the CUDA kernel
    module = SourceModule("""
        __global__ void binary_to_grayscale(const unsigned char *input, unsigned char *output, int size)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size)
            {
                output[idx] = input[idx] == 1 ? 255 : 0;
            }
        }
    """)

    # Get the kernel function
    binary_to_grayscale_kernel = module.get_function("binary_to_grayscale")

    # Prepare the kernel launch
    block_size = 256
    grid_size = (output_size + block_size - 1) // block_size

    # Launch the kernel
    binary_to_grayscale_kernel(input_data,
                               output_data,
                               np.int32(output_size),
                               block=(block_size, 1, 1),
                               grid=(grid_size, 1))

    # Copy the result back to the host
    output_array = np.empty(output_size, dtype=np.uint8)
    cuda.memcpy_dtoh(output_array, output_data)

    # Clean up the CUDA context
    if context is not None:
        context.pop()

    return output_array



# Example usage
binary_fragments = [b'\x01\x02\x03', b'\xFF\x00\xFF', b'\x80\x7F\x81']
grayscale_array = gpu_binary_to_image(binary_fragments)
print(grayscale_array)


def fill_remaining_elements(pixelsArr, frame_height, frame_width):
    remaining_elements = frame_height * frame_width - len(pixelsArr)

    if remaining_elements == 0:
        return pixelsArr
    # Calculate the number of remaining elements needed, and fill the array with 0s
    # Define the elements to pad
    padding_elements = [255] * 8 + [0, 255]

    pixelsArr = np.append(pixelsArr, padding_elements)

    # Append the padding elements
    # Pad the array with zeros
    pixelsArr = np.pad(pixelsArr,
                       (0, (frame_height * frame_width) - len(pixelsArr)),
                       'constant',
                       constant_values=(0, 0))

    return pixelsArr


def save_frames_as_images(frames, output_folder):
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(f'{output_folder}/frame_{i}.png')
    return True
