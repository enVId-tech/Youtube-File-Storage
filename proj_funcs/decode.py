import cv2
import numpy as np
import time
import ecc.hamming_funcs as hamming
from constants import FRAME_HEIGHT, FRAME_WIDTH, OUTPUT_PATH, OUTPUT_FILE

def decode_video():
    try:
        timer = time.time()
        video = cv2.VideoCapture(f'./output_files/{OUTPUT_PATH}')

        if not video.isOpened():
            print(f"Error: Could not open video file at {OUTPUT_PATH}")
            return False

        bit_frames = []
        numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in range(numFrames):
            ret, frame = video.read()
            if not ret:
                break

            if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
                print(f"Error: Frame dimensions do not match ({frame.shape[0]}x{frame.shape[1]}).")
                exit(1)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temp_bit_frames = gray_frame.flatten()

            # Remove trailing zeros
            if temp_bit_frames.size % 8 != 0:
                print("Error: Bit frames size is not a multiple of 8.")
                exit(1)

            bit_frames.append(temp_bit_frames)

        if len(bit_frames) == 0:
            print("Error: No frames were read from the video.")
            exit(1)

        print(f"1dec. Number of frames read: {len(bit_frames)}")

        # Combine all bit frames into a single array
        bit_frames = np.concatenate(bit_frames).flatten()

        if not len(bit_frames) == (FRAME_HEIGHT * FRAME_WIDTH * numFrames):
            print(f"Error: Number of bits read ({len(bit_frames)}) does not match the expected number ({FRAME_HEIGHT * FRAME_WIDTH * numFrames}).")
            exit(1)

        print(f"2dec. Length of bit frames: {len(bit_frames)}")
        
        bit_frames = np.where(bit_frames > 127, 1, 0)

        # Remove trailing zeros
        bit_frames = np.trim_zeros(bit_frames, 'b')

        if bit_frames.size % 12 != 0:
            while bit_frames.size % 12 != 0:
                bit_frames = np.append(bit_frames, 0)

        if bit_frames.size % 12 != 0:
            print("Error: Bit frames size is not a multiple of 12.")
            exit(1)

        # Convert 255s and 0s to 1s and 0s
        print(f"3dec. Length of bit frames after conversion: {len(bit_frames)}")

        # Decode the bit frames (Bits are already in 8-bit format, so no need to split them into 8-bit chunks)
        bit_frames = np.array([
            hamming.decode(bit_frames[i:i + 12])
            for i in range(0, len(bit_frames), 12)
        ])

        bit_frames = np.concatenate([np.array(list(s), dtype=int) for s in bit_frames])

        # Convert binary strings to integers
        print(f"4dec. Length of bit frames after decoding: {len(bit_frames)}")

        # Save the bit frames to a file
        bit_frames = np.packbits(bit_frames)

        print(f"5dec. Length of bit frames after packing: {len(bit_frames)}")

        with open(f'./output_files/{OUTPUT_FILE}', 'wb') as file:
            file.write(bit_frames)

        print(f"6dec. File saved successfully! Time taken: {time.time() - timer} seconds.")
        return True
    except Exception as e:
        print(f"Error in undo_main(): {e}")
        print(f"Runtime error occurred at {time.time() - timer} seconds.")
        return False