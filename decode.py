import cv2
import numpy as np
from checksum import hamming_decode, calculate_checksum, remove_trailing_zeros
from constants import FRAME_HEIGHT, FRAME_WIDTH, OUTPUT_PATH, OUTPUT_FILE

def decode_video():
    try:
        video = cv2.VideoCapture(f'./output_files/{OUTPUT_PATH}')

        if not video.isOpened():
            print(f"Error: Could not open video file at {OUTPUT_PATH}")
            return False

        bit_frames = []
        checksums = []

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
                print(f"Error: Frame dimensions do not match ({frame.shape[0]}x{frame.shape[1]}).")
                exit(1)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temp_bit_frames = np.unpackbits(gray_frame.flatten())

            # Remove trailing zeros
            if temp_bit_frames.size % 8 != 0:
                print("Error: Bit frames size is not a multiple of 8.")
                exit(1)

            bit_frames.append(temp_bit_frames.astype(np.uint8))

            checksum = calculate_checksum(temp_bit_frames)
            checksums.append(checksum)

        if len(bit_frames) == 0:
            print("Error: No frames were read from the video.")
            return False

        # Combine all bit frames into a single array
        bit_frames = np.concatenate(bit_frames).flatten()

        # Remove trailing zeros
        bit_frames = remove_trailing_zeros(bit_frames, FRAME_HEIGHT * FRAME_WIDTH)

        if bit_frames.size % 7 != 0:
            while bit_frames.size % 7 != 0:
                bit_frames = np.append(bit_frames, 0)

            if bit_frames.size % 7 != 0:
                print("Error: Bit frames size is not a multiple of 7.")
                return False

        if (len(bit_frames) % 7) != 0:
            print("Error: Bit frames length is not a multiple of 7.")
            exit(1)

        # Hamming decode each 4-bit block
        bit_frames = np.array([
            hamming_decode(bit_frames[i:i + 7])
            for i in range(0, len(bit_frames), 7)
        ]).flatten()

        # Save the bit frames to a file
        bit_frames = np.packbits(bit_frames)

        with open(f'./output_files/{OUTPUT_FILE}', 'wb') as file:
            file.write(bit_frames)

        print("File saved successfully!")
        return True
    except Exception as e:
        print(f"Error in undo_main(): {e}")
        exit(1)