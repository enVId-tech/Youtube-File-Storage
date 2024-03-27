import cv2
import numpy as np
import ecc.hamming_codec as hamming
from constants import FRAME_HEIGHT, FRAME_WIDTH, FRAME_RATE, INPUT_PATH, OUTPUT_PATH

def encode_file():
    try:
        with open(f'./input_files/{INPUT_PATH}', 'rb') as file:
            binary_data = np.fromfile(file, dtype=np.uint8)
        
        print(binary_data)

        print("1enc. File Read Successfully!")

        binary_data = np.unpackbits(binary_data)

        print(f"2enc. Length of binary data: {len(binary_data)}")

        np.array([
            hamming.encode(binary_data[i:i + 8])
            for i in range(0, len(binary_data), 8)
        ])

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
            cv2.VideoWriter_fourcc(*'H264'),
            FRAME_RATE,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        for frame in frames:
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame)

        video_writer.release()

        print("Video saved successfully!")
        return True
    except Exception as e:
        print(f"Error in main(): {e}")
        exit(1)

# Pass in array as numpy array
