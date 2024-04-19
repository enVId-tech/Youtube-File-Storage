# Video Settings
DEVICE = 'cpu'
FRAME_RATE = 30
FRAME = {
    '4k': [2160, 4096],
    '1440p': [1440, 2560],
    '1080p': [1080, 1920],
    '720p': [720, 1280],
    '480p': [480, 854],
    '144p': [144, 256]
}
INPUT_PATH = 'bruh.zip'
OUTPUT_PATH = 'output.mp4'
OUTPUT_FILE = 'output_files.zip'
FRAME_HEIGHT, FRAME_WIDTH = FRAME['144p']

# Hamming Code Settings
PARITY_BLOCK_SIZE = 1024 # Is in bits, keep it as a power of 2