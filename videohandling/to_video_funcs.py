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

def file_convert_to_video(frame, input_file, output_file, frame_rate,
                          device):
    try:
        height = frame[0]
        width = frame[1]

        sys.set_int_max_str_digits(100000000)
        binary_fragments = file_to_binary_fragments(input_file)

        if device == 'gpu':
            img = gpu_binary_to_image(binary_fragments, MAX_MEMORY)
        else:
            img = cpu_binary_to_image(binary_fragments, MAX_MEMORY, MAX_THREADS)

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


def file_to_binary_fragments(file_path):
    with open(file_path, 'rb') as file:
        binary_string = file.read()
        fragments = []

        # Split binary string into fragments
        for i in range(0, len(binary_string), MAX_BINARY_SIZE):
            fragment = binary_string[i:i + MAX_BINARY_SIZE]
            fragments.append(fragment)

        print(f'Number of fragments: {len(fragments)}')

        return fragments


def convert_binary_to_grayscale(img, start_idx, end_idx):
    for i in range(start_idx, end_idx):
        img[i] = img[i] * 255


def cpu_binary_to_image(fragments, max_memory=1000000, num_threads=8):
    binary_string = ''.join(
        map(lambda x: bin(int.from_bytes(x, byteorder='big'))[2:], fragments))
    binary_array = np.array(list(map(int, binary_string)), dtype=np.uint8)
    image_array = binary_array.astype(np.uint8)

    block_size = len(image_array) // num_threads

    threads = []
    start_idx = 0
    memory_available = psutil.virtual_memory().available
    for i in range(num_threads):
        end_idx = min(start_idx + block_size, len(image_array))
        thread = threading.Thread(target=convert_binary_to_grayscale,
                                  args=(image_array.copy(), start_idx,
                                        end_idx))
        threads.append(thread)
        thread.start()
        memory_used = memory_available - psutil.virtual_memory().available
        start_idx = end_idx

        # Check memory usage and pause if necessary
        while memory_used > max_memory:
            time.sleep(1)
            memory_used = sum(
                [thread._tstate_lock for thread in threads if thread.is_alive()])

    for thread in threads:
        thread.join()

    return image_array

# MUST BE RUN ON A GPU-ENABLED MACHINE
# This function uses CUDA to convert the binary array to an image
# Note: This function MUST get a binary array as input
def gpu_binary_to_image(fragments, max_memory=1000000):
    # Import CUDA and create the kernel function
    from pycuda import driver as drv
    from pycuda.compiler import SourceModule
    import pycuda.autoinit

    mod = SourceModule("""
    __global__ void convert_binary_to_grayscale(unsigned char *img, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = idx; i < size; i += blockDim.x * gridDim.x)
        {
            img[i] = img[i] * 255;
        }
    }
    """)

    binary_string = ''.join(
        map(lambda x: bin(int.from_bytes(x, byteorder='big'))[2:], fragments))
    binary_array = np.array(list(map(int, binary_string)), dtype=np.uint8)
    image_array = binary_array.astype(np.uint8)

    block_size = 256
    grid_size = (image_array.size + block_size - 1) // block_size

    func = mod.get_function("convert_binary_to_grayscale")

    # Allocate memory for the GPU data
    image_array_gpu = drv.mem_alloc(image_array.nbytes)
    drv.memcpy_htod(image_array_gpu, image_array)

    # Create CUDA stream
    stream = drv.Stream()

    # Event for synchronization
    event = drv.Event()

    processed_size = 0
    # Check memory usage
    mem_info = psutil.virtual_memory()

    while processed_size < image_array.size:
        original_mem_state = mem_info.available - mem_info.used
        if original_mem_state < max_memory:
            # Execute kernel
            func(image_array_gpu,
                 np.int32(image_array.size),
                 block=(block_size, 1, 1),
                 grid=(grid_size, 1),
                 stream=stream)

            # Synchronize and wait for the kernel to finish
            event.record()
            event.synchronize()

            # Copy the result back to host
            drv.memcpy_dtoh(image_array, image_array_gpu)

            # Update processed size
            processed_size = image_array.size
        else:
            time.sleep(1)

    return image_array


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
