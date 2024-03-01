import numpy as np
from image_funcs import toBin, fillRemainingElements, toImg

def textToImage(BASE, FRAME_HEIGHT, FRAME_WIDTH, text_file, image_file, device='gpu'):
    try:
        frame_pixels = np.array([])

        # Convert the text file to binary
        pixels = toBin(BASE, text_file, 'r')

        # Append the binary pixels to frame_pixels
        frame_pixels = np.append(frame_pixels, pixels)

        # Fill the remaining elements of frame_pixels
        frame_pixels = fillRemainingElements(frame_pixels, FRAME_HEIGHT, FRAME_WIDTH)

        # Print the length of pixels
        print(len(pixels))

        # Convert the binary pixels to an image and save it to a file
        img = toImg(frame_pixels, image_file, FRAME_HEIGHT, FRAME_WIDTH, device)

        if img is not None:
            return True
    except FileNotFoundError as e:
        print(e)
        return False
