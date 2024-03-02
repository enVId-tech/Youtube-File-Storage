def create_image(file_path, binaryList):
    # Save the binary list to a file
    with open(file_path, 'w') as f:
        f.write(bytes(binaryList))
    return True

