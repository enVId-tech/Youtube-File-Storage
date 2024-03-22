import os
import hashlib
import numpy as np

def compute_checksum(data):
    try:
        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'rb') as file:
                file_data = file.read()
        elif isinstance(data, bytes):
            file_data = data
        else:
            print(f"Error in compute_checksum: Input data is not a valid file path or byte array.")
            return None

        checksum = hashlib.md5(file_data).hexdigest()
        return checksum
    except Exception as e:
        print(f"Error in compute_checksum: {e}")
        return None

class HammingError(Exception):
    pass

def validate_data(data, length):
    if len(data) < length:
        raise HammingError(f"Data '{data}' is too short.")
    elif len(data) % length != 0:
        raise HammingError(f"Data '{data}' is not a multiple of {length}.")

def hamming_encode(data):
    try:
        validate_data(data, 4)
        d = np.array(data, dtype=np.uint8)

        # Calculate parity bits using bitwise operations and vectorization
        p1 = np.bitwise_xor.reduce(d[[0, 1, 3]], axis=0) & 1
        p2 = np.bitwise_xor.reduce(d[[0, 2, 3]], axis=0) & 1
        p3 = np.bitwise_xor.reduce(d[[1, 2, 3]], axis=0) & 1

        # Create the encoded data
        return np.array([p1, p2, d[0], p3, d[1], d[2], d[3]], dtype=np.uint8)
    except Exception as e:
        raise HammingError(f"Error in hamming_encode: {e}")

def hamming_decode(data):
    try:
        validate_data(data, 7)
        d = np.array(data, dtype=np.uint8)

        # Calculate parity bits using bitwise operations and vectorization
        p1 = (d[0] ^ d[2] ^ d[4] ^ d[6]) & 1
        p2 = (d[1] ^ d[2] ^ d[5] ^ d[6]) & 1
        p3 = (d[3] ^ d[4] ^ d[5] ^ d[6]) & 1

        # Calculate error position using a lookup table
        error_pos = [0, 0, 0, 1, 0, 2, 3, 0][p3 * 4 + p2 * 2 + p1]

        # Flip the bit at error_pos if necessary
        if error_pos:
            d[error_pos - 1] ^= 1

        return d[[2, 4, 5, 6]]
    except Exception as e:
        raise HammingError(f"Error in hamming_decode: {e}")
    
def calculate_checksum(data):
    return np.sum(data)
