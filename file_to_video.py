from videohandling.to_video_funcs import file_convert_to_video

# Frame dimensions (height, width)
FRAME = {
    '4k': [2160, 4096],
    '1440p': [1440, 2560],
    '1080p': [1080, 1920],
    '720p': [720, 1280],
    '480p': [480, 854],
    '144p': [144, 256]
}

DEVICE = 'cpu'
FRAME_RATE = 60

INPUT_FILE = 'bruh.mp4'
OUTPUT_VIDEO = 'outputvideo.mp4'


def main():
    # Convert the text file to an image
    txtToImg = file_convert_to_video(FRAME['1080p'],
                                     f'./input_files/{INPUT_FILE}',
                                     f'output_files/{OUTPUT_VIDEO}',
                                     FRAME_RATE, DEVICE)

    if txtToImg:
        print(
            f'{INPUT_FILE.split(".")[-1].upper()} file converted to {OUTPUT_VIDEO.split(".")[-1].upper()} successfully.'
        )
        exit(0)
    else:
        print(
            f'{INPUT_FILE.split(".")[-1].upper()} file conversion to {OUTPUT_VIDEO.split(".")[-1].upper()} failed.'
        )
        exit(1)


if __name__ == "__main__":
    main()
