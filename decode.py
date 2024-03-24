import cv2
import numpy as np
from checksum import remove_trailing_zeros
# from hammingenc import hamming_decode as decode
from ldpc import decode
from constants import FRAME_HEIGHT, FRAME_WIDTH, OUTPUT_PATH, OUTPUT_FILE

def decode_video():
    try:
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

        # Remove trailing zeros
        bit_frames = remove_trailing_zeros(bit_frames, FRAME_HEIGHT * FRAME_WIDTH)

        if bit_frames.size % 7 != 0:
            while bit_frames.size % 7 != 0:
                bit_frames = np.append(bit_frames, 0)

            if bit_frames.size % 7 != 0:
                print("Error: Bit frames size is not a multiple of 7.")
                exit(1)

        print(f"2dec. Length of bit frames: {len(bit_frames)}")

        if (len(bit_frames) % 7) != 0:
            print("Error: Bit frames length is not a multiple of 7.")
            exit(1)

        # Decode the bit frames using Hamming code
        bit_frames = decode(bit_frames)
        
        # Save the bit frames to a file
        bit_frames = np.packbits(bit_frames)

        with open(f'./output_files/{OUTPUT_FILE}', 'wb') as file:
            file.write(bit_frames)

        print("File saved successfully!")
        return True
    except Exception as e:
        print(f"Error in undo_main(): {e}")
        exit(1)