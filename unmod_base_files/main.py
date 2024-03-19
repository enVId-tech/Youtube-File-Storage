import cv2
import numpy as np
import hashlib
import os

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
FRAME_HEIGHT, FRAME_WIDTH = FRAME['4k']

def compute_checksum(data):
    try:
        if os.path.exists(data):
            with open(data, 'rb') as file:
                file_data = file.read()
                checksum = hashlib.md5(file_data).hexdigest()
                return checksum
        else:
            print(f"Error in compute_checksum: File '{data}' does not exist.")
            return None
    except Exception as e:
        print(f"Error in compute_checksum: {e}")
        return None


def hamming_encode(data):
    try:
        # 4-bit data
        d = np.array(data, dtype=np.uint8)

        # Parity bits
        p1 = d[0] ^ d[1] ^ d[3]
        p2 = d[0] ^ d[2] ^ d[3]
        p3 = d[1] ^ d[2] ^ d[3]

        # Encoded data
        d_encoded = np.array([p1, p2, d[0], p3, d[1], d[2], d[3]], dtype=np.uint8)
        return d_encoded
    except Exception as e:
        print(f"Error in hamming_encode: {e}")
        exit(1)


def hamming_decode(data):
    try:
        # 7-bit data
        d = np.array(data, dtype=np.uint8)

        # Parity bits
        p1 = d[0]
        p2 = d[1]
        p3 = d[3]

        # Error correction
        e1 = p1 ^ d[2] ^ d[4] ^ d[6]
        e2 = p2 ^ d[2] ^ d[5] ^ d[6]
        e3 = p3 ^ d[4] ^ d[5] ^ d[6]

        # Error detection
        error = e1 * 4 + e2 * 2 + e3

        if error != 0:
            d[error - 1] = 1 - d[error - 1]

        # Decoded data
        d_decoded = np.array([d[2], d[4], d[5], d[6]], dtype=np.uint8)
        return d_decoded
    except Exception as e:
        print(f"Error in hamming_decode: {e}")
        exit(1)

def calculate_checksum(data):
    return np.sum(data)

def remove_trailing_zeros(array, padding_length):
    array_length = len(array)
    num_trailing_zeros = padding_length
    trimmed_array = array[:array_length - num_trailing_zeros]
    return trimmed_array


def main():
    try:
        with open(f'./input_files/{INPUT_PATH}', 'rb') as file:
            binary_data = np.fromfile(file, dtype=np.uint8)
        binary_data = np.unpackbits(binary_data)

        print(f"First 10 bytes of binary data: {binary_data[:10]}")

        binary_data = np.array([
            hamming_encode(binary_data[i:i + 4])
            for i in range(0, len(binary_data), 4)
        ]).flatten()

        remainder = binary_data.size % (FRAME_HEIGHT * FRAME_WIDTH)
        if remainder != 0:
            padding_size = FRAME_HEIGHT * FRAME_WIDTH - remainder
            binary_data = np.pad(binary_data, (0, padding_size),
                                 mode='constant',
                                 constant_values=0)

        frames = binary_data.reshape(-1, FRAME_HEIGHT, FRAME_WIDTH) * 255

        for i, frame in enumerate(frames):
            cv2.imwrite(f'./output_files/frames/frame_{i}.png', frame)

        video_writer = cv2.VideoWriter(f'./output_files/{OUTPUT_PATH}',
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT),
                                       isColor=False)

        for frame in frames:
            video_writer.write(frame.astype(np.uint8))

        video_writer.release()
        print("Video saved successfully!")
        return True
    except Exception as e:
        print(f"Error in main(): {e}")
        return False


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

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temp_bit_frames = np.where(gray_frame > 127, 1, 0)
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

        # Hamming decode each 4-bit block
        bit_frames = np.array([
            hamming_decode(bit_frames[i:i + 7])
            for i in range(0, len(bit_frames), 7)
        ]).flatten()

        # Convert bit frames to bytes
        bit_frames = np.packbits(bit_frames)

        with open(f'./output_files/{OUTPUT_FILE}', 'wb') as file:
            file.write(bit_frames)

        print("File saved successfully!")
        return True
    except Exception as e:
        print(f"Error in undo_main(): {e}")
        return False


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
