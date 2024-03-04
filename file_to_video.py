from videohandling.to_video_funcs import file_convert_to_video

# Frame dimensions (height, width)
FRAME_4K = [2160, 4096]
FRAME_1440P = [1440, 2560]
FRAME_1080P = [1080, 1920]
FRAME_720P = [720, 1280]
FRAME_480P = [480, 854]
FRAME_144P = [144, 256]

DEVICE = 'cpu'
FRAME_RATE = 60

def main():
    # Convert the text file to an image
    txtToImg = file_convert_to_video(FRAME_144P[0], FRAME_144P[1],
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
