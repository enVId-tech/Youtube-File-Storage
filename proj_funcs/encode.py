# nom nom
import cv2
import numpy as np
import ecc.hamming_funcs as hamming
import time
import math
from tqdm import tqdm
from constants import FRAME_HEIGHT, FRAME_WIDTH, FRAME_RATE, INPUT_PATH, OUTPUT_PATH, PARITY_BLOCK_SIZE


def encode_file():
    try:
        timer = time.time()
        with open(f'./input_files/{INPUT_PATH}', 'rb') as file:
            binary_data = np.fromfile(file, dtype=np.uint8)

        # print("1enc. File Read Successfully!")

        binary_data = np.unpackbits(binary_data)

        # print(f"2enc. Length of binary data: {len(binary_data)}")

        binary_data_len = len(binary_data)

        byte_size = binary_data_len / 8

        if not byte_size % 8 == 0:
            return Exception("Byte data is not a multiple of 8")

        block_size = PARITY_BLOCK_SIZE

        byte_sizes = {}

        for exponent in range(3, int(math.log2(block_size)) + 1):
            key = str(2 ** exponent)
            byte_sizes[key] = 0

        remainder = byte_size % block_size

        byte_sizes[str(block_size)] += int((byte_size - remainder) / block_size)

        if remainder > 0:
            while remainder > 0:
                print(remainder, block_size)
                if remainder < block_size and block_size >= 8:
                    byte_sizes[str(int(block_size))] += 0
                    block_size /= 2
                else:
                    byte_sizes[str(int(block_size))] = int(block_size / (remainder - (remainder % block_size)))
                    remainder %= block_size

        print(binary_data_len, byte_size, block_size, byte_sizes, remainder)
        exit(1)

        # Encode the binary data using Hamming(8, 4) code
        print("Encoding binary data using Hamming(8, 4) code...\n Progress: ")
        binary_data = np.array([
            hamming.encode(binary_data[i:i + 8])
            for i in tqdm(range(0, len(binary_data), 8))
        ])

        binary_data = np.concatenate(
            [np.array(list(s), dtype=int) for s in binary_data])

        # print(f"3enc. Length of binary data after encoding: {len(binary_data)}")

        # Convert 1s and 0s to 255s and 0s
        binary_data = np.where(binary_data == 1, 255, 0)

        remainder = len(binary_data) % (FRAME_HEIGHT * FRAME_WIDTH)

        if remainder != 0:
            padding_length = (FRAME_HEIGHT * FRAME_WIDTH) - remainder
            binary_data = np.append(binary_data, np.zeros(padding_length))

        # print(f"4enc. Length of binary data after padding: {len(binary_data)}")

        # Split the binary data into frames
        print(f"Splitting binary data into frames...\n Progress: ")
        frames = np.array([
            binary_data[i:i + (FRAME_HEIGHT * FRAME_WIDTH)] for i in tqdm(
                range(0, len(binary_data), FRAME_HEIGHT * FRAME_WIDTH))
        ])

        # Convert the frames to 8-bit grayscale images
        frames = np.array([
            frame.reshape(FRAME_HEIGHT, FRAME_WIDTH).astype(np.uint8)
            for frame in frames
        ])

        # print(f"5enc. Frames created successfully! Number of frames: {len(frames)}")

        # Save the frames to a video file
        video_writer = cv2.VideoWriter(f'./output_files/{OUTPUT_PATH}',
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))

        for frame in frames:
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame)

        video_writer.release()

        print(
            f"\nVideo file saved successfully! Time taken: {time.time() - timer} seconds"
        )
        return True
    except Exception as e:
        print(f"Error in main(): {e}")
        print(f"Runtime error occurred at ${time.time() - timer} seconds")
        exit(1)
