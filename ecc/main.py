from hamming_codec import calcRedundantBits, posRedundantBits, calcParityBits, detectError, introduceError, correctError, removeRedundantBits
import time


def main():
    def test_hamming_code(test_case_num, org_data, error):
        timer = time.time()

        print(f"----- Test case {test_case_num} -----")
        print("Original data:", org_data)
        
        m = len(org_data)
        r = calcRedundantBits(m)
        print("No of redundant bits:", r)

        arr = posRedundantBits(org_data, r)
        arr = calcParityBits(arr, r)
        print("Encoded data:", arr)
        
        code = introduceError(arr, error)
        print("Data with error:", code)

        correction = detectError(code, r)
        print("Error at position:", correction)

        corrected = correctError(code, correction)
        print("Corrected data:", corrected)
        
        data = removeRedundantBits(corrected)
        print("Removed redundant bits:", data)

        print(
            f"\nTest case {test_case_num} {'PASSED' if org_data == data else 'FAILED'}\n"
        )

        print("--- %s seconds ---" % (time.time() - timer))

    # Test case 1 (4 bits, 3 parity bits, 1 error)
    # Maximum number of data bits that can be encoded with 3 parity bits is 4
    test_hamming_code(1, '0110', True)

    # Test case 2 (8 bits, 4 parity bits, no errors)
    # Maximum number of data bits that can be encoded with 4 parity bits is 11
    test_hamming_code(2, '10110011', False)

    # Test case 3 (16 bits, 4 parity bits, 1 error)
    # Maximum number of data bits that can be encoded with 4 parity bits is 11
    test_hamming_code(3, '1011001101010011', True)

    # Test case 4 (32 bits, 4 parity bits, 2 errors)
    # Maximum number of data bits that can be encoded with 4 parity bits is 11
    test_hamming_code(4, '10110011010100111011001101010011', True)

    # Test case 5 (64 bits, 6 parity bits, 3 errors)
    # Maximum number of data bits that can be encoded with 6 parity bits is 26
    test_hamming_code(5, '1011001101010011101100110101001110110011010100111011001101010011', True)


if __name__ == '__main__':
    main()
