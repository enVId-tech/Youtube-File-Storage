import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2

# Define the maximum size of each binary fragment
MAX_BINARY_SIZE = 1000000  # Adjust this value as needed

def file_convert_to_video(height, width, input_file, output_file, frame_rate,
                          device):
    try:
        sys.set_int_max_str_digits(100000000)
        binary_fragments = file_to_binary_fragments(input_file)

        if device == 'gpu':
            img = gpu_binary_to_image(binary_fragments)
        else:
            img = cpu_binary_to_image(binary_fragments)

        print(f'Image shape: {img[0:100]}')

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
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec
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
        


def cpu_binary_to_image(fragments):
    # Using multiple threads to convert the pixels to an image using distributed computing
    with ThreadPoolExecutor() as executor:
        binary_string = ''.join(
            map(lambda x: bin(int.from_bytes(x, byteorder='big'))[2:],
                fragments))
        return np.array(list(executor.map(int, binary_string)), dtype=np.uint8)


# MUST BE RUN ON A GPU-ENABLED MACHINE
# This function uses CUDA to convert the binary array to an image
# Note: This function MUST get a binary array as input
def gpu_binary_to_image(fragments):
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    # Define CUDA function
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
    # Assuming 'binary_array' is your binary array
    binary_array = np.array(list(map(int, binary_string)), dtype=np.uint8)

    # Convert binary array to uint8
    image_array = binary_array.astype(np.uint8)

    # Get function
    func = mod.get_function("convert_binary_to_grayscale")

    # Define block and grid sizes
    block_size = 256
    grid_size = (image_array.size + block_size - 1) // block_size

    # Create CUDA stream
    stream = drv.Stream()

    # Allocate memory on the device
    image_array_gpu = drv.mem_alloc(image_array.nbytes)

    # Copy the data to the device
    drv.memcpy_htod_async(image_array_gpu, image_array, stream)

    # Call function on GPU
    func(image_array_gpu,
         np.int32(image_array.size),
         block=(block_size, 1, 1),
         grid=(grid_size, 1),
         stream=stream)

    # Copy the data back to the host
    drv.memcpy_dtoh_async(image_array, image_array_gpu, stream)

    # Wait for all operations to finish
    stream.synchronize()

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
