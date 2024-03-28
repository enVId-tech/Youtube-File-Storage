import zlib

def compute_checksum(file_path):
    try:
        with open(file_path, 'rb') as file:
            return zlib.crc32(file.read())
    except Exception as e:
        print(f"Error in compute_checksum(): {e}")
        return 0