# Import libraries to process to hex
import binascii
import cv2
import numpy as np

# Global
base = 16
frameHeight = 1080
frameWidth = 1920

def main():
    binaryArray = np.array([])

    # Open the file to read a text file to hex
    with open('text.txt', 'r') as file:
        # Read the file
        data = file.read()
        # Convert the file to hex
        hex_data = binascii.hexlify(data.encode())
        # Print the hex data
        print(hex_data)
        # Hex to binary
        print(bin(int(hex_data, base)))
        # Append the binary data to the array as a 2D array size of the frame, inserting each binary value as its own element
        binaryArray = np.append(binaryArray, bin(int(hex_data, base)))

        print(binaryArray)

        # # Open the file to write the hex data
        # with open('hex.txt', 'w') as file:
        #     # Write the hex data converted back to readable text
        #     file.write(binascii.unhexlify(hex_data).decode())

        binaryArray = binaryArray.reshape(frameHeight, frameWidth)

    

    

if __name__ == "__main__":
    main()
