import concurrent.futures
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

from ecc.hamming_funcs import calcParityBits, calcRedundantBits, detectError, correctError, posRedundantBits, removeRedundantBits

def encode_parallel(arr):
    if type(arr) == np.ndarray:
        arr = arr.tolist()
    str_arr = ''.join([str(elem) for elem in arr])
    m = len(str_arr)
    r = calcRedundantBits(m)
    arr = posRedundantBits(str_arr, r)
    arr = calcParityBits(arr, r)
    return arr

def decode_parallel(arr):
    if type(arr) == np.ndarray:
        arr = arr.tolist()
    str_arr = ''.join([str(elem) for elem in arr])
    m = 8
    r = calcRedundantBits(m)
    correction = detectError(str_arr, r)
    corrected = correctError(str_arr, correction)
    data = removeRedundantBits(corrected)
    return data

def encode_parallel_wrapper(data):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(encode_parallel, data)
    return list(results)

def decode_parallel_wrapper(data):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(decode_parallel, data)
    return list(results)

def calcParityBits_gpu(arr, r):
    arr_gpu = gpuarray.to_gpu(arr)
    n = len(arr)
    res_gpu = gpuarray.zeros(n, dtype=np.int32)

    # Define the CUDA kernel
    kernel = cuda.MemoryPointer(cuda.src_module, "kernel")

    # Launch the CUDA kernel
    block_size = 1024
    grid_size = (n + block_size - 1) // block_size
    kernel(arr_gpu, res_gpu, np.int32(n), np.int32(r), block=(block_size, 1, 1), grid=(grid_size, 1))

    res = res_gpu.get()
    return res

def encode_parallel_gpu(arr):
    if type(arr) == np.ndarray:
        arr = arr.tolist()
    str_arr = ''.join([str(elem) for elem in arr])
    m = len(str_arr)
    r = calcRedundantBits(m)
    arr = posRedundantBits(str_arr, r)
    arr = calcParityBits_gpu(arr, r)
    return arr

def decode_parallel_gpu(arr):
    if type(arr) == np.ndarray:
        arr = arr.tolist()
    str_arr = ''.join([str(elem) for elem in arr])
    m = 8
    r = calcRedundantBits(m)
    correction = detectError(str_arr, r)
    corrected = correctError(str_arr, correction)
    data = removeRedundantBits(corrected)
    return data