import numpy as np
from numba import cuda, njit

@cuda.jit
def hamming_encode_kernel(data, code, n, r):
    """
    CUDA kernel for Hamming encoding.
    """
    i = cuda.grid(1)
    if i < n:
        code[i + r] = data[i]

    if i < r:
        pos = 2 ** i
        for j in range(pos - 1, n + r, 2 * pos):
            start = j
            end = min(j + pos, n + r)
            for k in range(start, end):
                code[i] ^= code[k]

@cuda.jit
def hamming_decode_kernel(code, decoded, n, r):
    """
    CUDA kernel for Hamming decoding and error correction.
    """
    i = cuda.grid(1)
    if i < n - r:
        decoded[i] = code[i + r]

    if i < r:
        pos = 2 ** i
        syndrome = 0
        for j in range(pos - 1, n, 2 * pos):
            start = j
            end = min(j + pos, n)
            for k in range(start, end):
                syndrome ^= code[k]

        if syndrome != 0:
            binary_str = int_to_binary(syndrome, r)
            error_position = 0
            for pos, val in enumerate(binary_str[::-1]):
                if val == '1':
                    error_position += 2 ** pos

            if error_position > 0:
                code[error_position - 1] ^= 1

@njit
def int_to_binary(num, length):
    """
    Helper function to convert an integer to a binary string.
    """
    binary = 0
    for _ in range(length):
        binary = (binary << 1) | (num & 1)
        num >>= 1

    binary_str = ""
    for i in range(length):
        binary_str = str(binary & 1) + binary_str
        binary >>= 1

    return binary_str

def hamming_encode(data):
    """
    Encodes the given data using the Hamming code (with CUDA parallelization).
    """
    n = len(data)
    r = 1
    while (2 ** r - r - 1) < n:
        r += 1

    code = np.zeros(n + r, dtype=np.int32)
    data_gpu = cuda.to_device(np.array(data, dtype=np.int32))
    code_gpu = cuda.to_device(code)

    threads_per_block = 32
    blocks_per_grid = (n + r + threads_per_block - 1) // threads_per_block

    hamming_encode_kernel[blocks_per_grid, threads_per_block](data_gpu, code_gpu, n, r)

    code = code_gpu.copy_to_host()
    return code

def hamming_decode(code):
    """
    Decodes the given Hamming code and corrects a single bit error (with CUDA parallelization).
    """
    n = len(code)
    r = 1
    while (2 ** r - r - 1) < n - r:
        r += 1

    decoded = np.zeros(n - r, dtype=np.int32)
    code_gpu = cuda.to_device(np.array(code, dtype=np.int32))
    decoded_gpu = cuda.to_device(decoded)

    threads_per_block = 32
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    hamming_decode_kernel[blocks_per_grid, threads_per_block](code_gpu, decoded_gpu, n, r)

    decoded = decoded_gpu.copy_to_host()
    return decoded