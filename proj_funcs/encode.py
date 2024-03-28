import cv2
import numpy as np
import ecc.hamming_funcs as hamming
# import ecc.hamming_codec as main_hamming
import ecc.optimized_funcs as opt_hamming
import time
from constants import FRAME_HEIGHT, FRAME_WIDTH, FRAME_RATE, INPUT_PATH, OUTPUT_PATH
import concurrent.futures

def encode_chunk(chunk):
    # Starts at 3 parity bits for 4 data bits
    # Increases by 1 parity bit for every next doubling of data bits\
    # Max should be 1024 data bits and 11 parity bits
    return np.array([opt_hamming.encode_parallel(chunk[i:i+8]) for i in range(0, len(chunk), 8)])

def encode_file():
    try:
        timer = time.time()
        with open(f'./input_files/{INPUT_PATH}', 'rb') as file:
            binary_data = np.fromfile(file, dtype=np.uint8)

        print("1enc. File Read Successfully!")

        binary_data = np.unpackbits(binary_data)

        print(f"2enc. Length of binary data: {len(binary_data)}")

        # Encode the binary data using Hamming(8, 4) code
        with concurrent.futures.ThreadPoolExecutor() as executor:
            encoded_chunks = executor.map(encode_chunk, np.array_split(binary_data, len(binary_data)//(8*FRAME_HEIGHT*FRAME_WIDTH)))

        binary_data = np.concatenate([np.array(list(s), dtype=int) for chunk in encoded_chunks for s in chunk])

        print(f"3enc. Length of binary data after encoding: {len(binary_data)}")

        # Convert 1s and 0s to 255s and 0s
        binary_data = np.where(binary_data == 1, 255, 0)

        remainder = len(binary_data) % (FRAME_HEIGHT * FRAME_WIDTH)

        if remainder != 0:
            padding_length = (FRAME_HEIGHT * FRAME_WIDTH) - remainder
            binary_data = np.append(binary_data, np.zeros(padding_length))

        print(f"4enc. Length of binary data after padding: {len(binary_data)}")

        # Split the binary data into frames
        frames = np.array([
            binary_data[i:i + (FRAME_HEIGHT * FRAME_WIDTH)]
            for i in range(0, len(binary_data), FRAME_HEIGHT * FRAME_WIDTH)
        ])

        # Convert the frames to 8-bit grayscale images
        frames = np.array([
            frame.reshape(FRAME_HEIGHT, FRAME_WIDTH).astype(np.uint8)
            for frame in frames
        ])

        print(f"5enc. Frames created successfully! Number of frames: {len(frames)}")

        # Save the frames to a video file
        video_writer = cv2.VideoWriter(
            f'./output_files/{OUTPUT_PATH}',
            cv2.VideoWriter_fourcc(*'mp4v'),
            FRAME_RATE,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        for frame in frames:
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame)

        video_writer.release()

        print(f"6enc. Video file saved successfully! Time taken: {time.time() - timer} seconds")
        return True
    except Exception as e:
        print(f"Error in main(): {e}")
        print(f"Runtime error occurred at ${time.time() - timer} seconds")
        return False
