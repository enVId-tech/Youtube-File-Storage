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

        # Syndrome bits
        s1 = d[0] ^ d[2] ^ d[4] ^ d[6]
        s2 = d[1] ^ d[2] ^ d[5] ^ d[6]
        s3 = d[3] ^ d[4] ^ d[5] ^ d[6]

        # Error correction
        e = s1 * 1 + s2 * 2 + s3 * 4
        if e != 0:
            d[e - 1] = 1 - d[e - 1]

        # Decoded data
        d_decoded = np.array([d[2], d[4], d[5], d[6]], dtype=np.uint8)
        return d_decoded
    except Exception as e:
        print(f"Error in hamming_decode: {e}")
        exit(1)

def calculate_checksum(data):
    return np.sum(data)


def add_parity_bits(data, parity_type='even'):
    if parity_type == 'even':
        parity_bit = np.sum(data) % 2
    elif parity_type == 'odd':
        parity_bit = 1 - (np.sum(data) % 2)
    else:
        raise ValueError("Invalid parity type. Use 'even' or 'odd'.")
    data_with_parity = np.append(data, parity_bit)
    return data_with_parity


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

        bit_frames = np.concatenate(bit_frames)

        # Hamming decode each 4-bit block
        bit_frames = np.array([
            hamming_decode(bit_frames[i:i + 7])
            for i in range(0, len(bit_frames), 7)
        ]).flatten()

        parity_bit = bit_frames[-1]
        bit_frames = bit_frames[:-1]
        if not np.any(np.sum(bit_frames) % 2 != parity_bit):
            print("Error: Parity check failed!")
            return False

        for checksum in checksums:
            calculated_checksum = calculate_checksum(bit_frames)
            if np.all(calculated_checksum != checksum):
                print("Error: Checksum mismatch detected!")
                return False

        bit_frames = bit_frames.tobytes()

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
