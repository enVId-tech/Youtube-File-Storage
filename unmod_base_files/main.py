import cv2
import numpy as np
from checksum import compute_checksum, hamming_encode, hamming_decode, calculate_checksum

DEVICE = 'cpu'
FRAME_RATE = 60
FRAME = {
    '4k': [2160, 4096],
    '1440p': [1440, 2560],
    '1080p': [1080, 1920],
    '720p': [720, 1280],
    '480p': [480, 854],
    '144p': [144, 256]
}
INPUT_PATH = 'input.txt'
OUTPUT_PATH = 'output.mp4'
OUTPUT_FILE = 'output.txt'
FRAME_HEIGHT, FRAME_WIDTH = FRAME['144p']

def remove_trailing_zeros(array, padding_length):
    try:
        if padding_length == 0:
            return array

        while padding_length > 0:
            if array[-1] == 0:
                array = array[:-1]
                padding_length -= 1
            else:
                break

        return array
    except Exception as e:
        print(f"Error in remove_trailing_zeros: {e}")
        exit(1)

def main():
    try:
        with open(f'./input_files/{INPUT_PATH}', 'rb') as file:
            binary_data = np.fromfile(file, dtype=np.uint8)
        binary_data = np.unpackbits(binary_data)

        print(f"First 10 bytes of binary data: {binary_data[:10]}")

        binary_data = hamming_encode(binary_data)

        remainder = len(binary_data) % (FRAME_HEIGHT * FRAME_WIDTH)
        if remainder != 0:
            padding_length = (FRAME_HEIGHT * FRAME_WIDTH) - remainder
            binary_data = np.append(binary_data, np.zeros(padding_length))

        print(f"Length of binary data: {len(binary_data)}")

        # Split the binary data into frames
        frames = np.array([
            binary_data[i:i + (FRAME_HEIGHT * FRAME_WIDTH)]
            for i in range(0, len(binary_data), FRAME_HEIGHT * FRAME_WIDTH)
        ])

        print(f"Amount of frames: {len(frames)}")

        # Convert the frames to 8-bit grayscale images
        frames = np.array([
            frame.reshape(FRAME_HEIGHT, FRAME_WIDTH).astype(np.uint8)
            for frame in frames
        ])

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
        print("Video saved successfully!")
        return True
    except Exception as e:
        print(f"Error in main(): {e}")
        exit(1)


def undo_main():
    try:
        video = cv2.VideoCapture(f'./output_files/{OUTPUT_PATH}')

        if not video.isOpened():
            print(f"Error: Could not open video file at {OUTPUT_PATH}")
            return False

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Amount of frames: {frame_count}")

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
            if temp_bit_frames.size % 8 != 0:
                print("Error: Bit frames size is not a multiple of 8.")
                exit(1)

            if not temp_bit_frames.size == FRAME_HEIGHT * FRAME_WIDTH:
                print(f"Error: Frame size is not {FRAME_HEIGHT}x{FRAME_WIDTH}.")
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

        print(f"Length of bit frames: {len(bit_frames)}")
        if (len(bit_frames) % 7) != 0:
            print("Error: Bit frames length is not a multiple of 7.")
            exit(1)

        # Hamming decode each 4-bit block
        bit_frames = np.array([
            hamming_decode(bit_frames[i:i + 7])
            for i in range(0, len(bit_frames), 7)
        ]).flatten()

        print(f"Length of bit frames after decoding: {len(bit_frames)}")

        # Save the bit frames to a file
        bit_frames = np.packbits(bit_frames)

        with open(f'./output_files/{OUTPUT_FILE}', 'wb') as file:
            file.write(bit_frames)

        print("File saved successfully!")
        return True
    except Exception as e:
        print(f"Error in undo_main(): {e}")
        exit(1)


if __name__ == "__main__":
    try:
        print("Running main...")
        main()
        print("Running undo_main...")
        undo_main()

        input_checksum = compute_checksum(f'./input_files/{INPUT_PATH}')
        output_checksum = compute_checksum(f'./output_files/{OUTPUT_FILE}')

        if input_checksum and output_checksum:
            if input_checksum == output_checksum:
                print("Checksums match!")
            else:
                print("Checksums do not match!")
        else:
            print("Checksums could not be computed!")

        print(f"Input checksum: {input_checksum}")
        print(f"Output checksum: {output_checksum}")

        print("Done!")
        exit(0)
    except Exception as e:
        print(f"Error in __main__: {e}")
        exit(1)
