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

def main():
    # Convert the text file to an image
    txtToImg = file_convert_to_video(FRAME['1080p'],
                                     './input_files/text.txt',
                                     'output_files/mainvideo.mp4', FRAME_RATE,
                                     DEVICE)

    if txtToImg:
        print('Text file converted to image successfully')
    else:
        print('Text file conversion to image failed')
    exit(0)

if __name__ == "__main__":
    main()
