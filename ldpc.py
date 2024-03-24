import numpy as np
from numba import cuda, int32, float32

# LDPC code parameters
RATE = 0.5  # Code rate (K/N)
MAX_ITERATIONS = 100  # Maximum number of decoding iteration

# Helper functions
@cuda.jit
def calculate_parity(msg, H, parity):
    row = cuda.grid(1)
    if row < H.shape[0]:
        temp_parity = 0
        for col in range(H.shape[1]):
            if H[row, col] == 1:
                temp_parity ^= msg[col]
        parity[row] = temp_parity

@cuda.jit
def update_codeword(codeword, syndrome, H):
    row = cuda.grid(1)
    if row < codeword.shape[0]:
        temp_syndrome = 0
        for col in range(H.shape[0]):
            if H[col, row] == 1:
                temp_syndrome ^= syndrome[col]
        if temp_syndrome == 1:
            codeword[row] ^= 1

# Encoding function
def encode(msg):
    K = len(msg)
    N = int(K / RATE)
    M = N - K

    H = np.random.randint(0, 2, (M, N), dtype=np.int32)

    codeword = np.zeros(N, dtype=np.int32)
    codeword[:K] = msg

    parity = np.zeros(M, dtype=np.int32)
    calculate_parity[M, 1](np.asarray(codeword), H, parity)

    codeword[K:] = parity

    return codeword

# Decoding function
def decode(received):
    N = len(received)
    K = int(N * RATE)
    M = N - K

    H = np.random.randint(0, 2, (M, N), dtype=np.int32)

    codeword = received.copy()
    syndrome = np.zeros(M, dtype=np.int32)

    for _ in range(MAX_ITERATIONS):
        calculate_parity[M, 1](np.asarray(codeword), H, syndrome)
        if np.all(syndrome == 0):
            break
        update_codeword[N, 1](np.asarray(codeword), np.asarray(syndrome), H)

    return codeword[:K]