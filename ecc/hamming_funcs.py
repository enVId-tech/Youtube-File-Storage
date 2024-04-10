import random
import numpy as np

def calcRedundantBits(m):
	try:
		# Use the formula 2 ^ r >= m + r + 1
		# to calculate the no of redundant bits.
		# Iterate over 0 .. m and return the value
		# that satisfies the equation

		for i in range(m):
			if(2**i >= m + i + 1):
				return i
			
		return 0
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)

def posRedundantBits(data, r):
	try:
		# Redundancy bits are placed at the positions
		# which correspond to the power of 2.
		j = 0
		k = 1
		m = len(data)
		res = ''

		# If position is power of 2 then insert '0'
		# Else append the data
		for i in range(1, m + r+1):
			if(i == 2**j):
				res = res + '0'
				j += 1
			else:
				res = res + data[-1 * k]
				k += 1

		# The result is reversed since positions are
		# counted backwards. (m + r+1 ... 1)
		return res[::-1]
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)

def calcParityBits(arr, r):
	try:
		n = len(arr)

		# For finding rth parity bit, iterate over
		# 0 to r - 1
		for i in range(r):
			val = 0
			for j in range(1, n + 1):

				# If position has 1 in ith significant
				# position then Bitwise OR the array value
				# to find parity bit value.
				if(j & (2**i) == (2**i)):
					val = val ^ int(arr[-1 * j])
					# -1 * j is given since array is reversed

			# String Concatenation
			# (0 to n - 2^r) + parity bit + (n - 2^r + 1 to n)
			arr = arr[:n-(2**i)] + str(val) + arr[n-(2**i)+1:]
		return arr
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)


def detectError(arr, nr):
	try:
		n = len(arr)
		res = 0

		# Calculate parity bits again
		for i in range(nr):
			val = 0
			for j in range(1, n + 1):
				if(j & (2**i) == (2**i)):
					val = val ^ int(arr[-1 * j])

			# Create a binary no by appending
			# parity bits together.

			res = res + val*(10**i)

		# Convert binary to decimal
		return int(str(res), 2)
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)

def introduceError(arr, error):
	try:
		if not error:
			return arr
		n = len(arr)
		# Generate a random no less than
		# length of array
		val = random.randint(0, n-1)

		# Flip the bit
		arr = arr[:val] + ('1' if arr[val] == '0' else '0') + arr[val+1:]
		return arr
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)

def correctError(arr, correction):
	try:
		if not correction:
			return arr
		idx = len(arr) - correction
		arr = arr[:idx] + ('0' if arr[idx] == '1' else '1') + arr[idx + 1:]
		return arr
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)

def removeRedundantBits(arr):
	try:
		n = len(arr)
		res = ''
		for i in range(1, n + 1):
			if(i & (i - 1) != 0):
				res = res + arr[-1 * i]
		return res[::-1]
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)

def encode(arr):
	try:
		if type(arr) == np.ndarray:
			arr = arr.tolist()
		str_arr = ''.join([str(elem) for elem in arr])
		m = len(str_arr)
		r = calcRedundantBits(m)
		arr = posRedundantBits(str_arr, r)
		arr = calcParityBits(arr, r)
		return arr
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)
    
def decode(arr):
	try:
		if type(arr) == np.ndarray:
			arr = arr.tolist()
		str_arr = ''.join([str(elem) for elem in arr])
		m = 8
		r = calcRedundantBits(m)
		correction = detectError(str_arr, r)
		corrected = correctError(str_arr, correction)
		data = removeRedundantBits(corrected)
		return data
	except Exception as e:
		print(f"Error in main(): {e}")
		exit(1)