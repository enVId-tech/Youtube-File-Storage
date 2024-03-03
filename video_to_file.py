from filehandling.to_file_funcs import video_convert_to_file

# Frame dimensions (height, width)
FRAME_4K = [2160, 4096]
FRAME_1440P = [1440, 2560]
FRAME_1080P = [1080, 1920]
FRAME_720P = [720, 1280]

DEVICE = 'gpu'
FRAME_RATE = 60

def main():
    # Convert the video to a text file
    vidToTxt = video_convert_to_file('./input_files/mainvideo.mp4',
                                     './output_files/javatest.jar', DEVICE)
    if vidToTxt:
        print('Video converted to text file successfully')
    else:
        print('Video conversion to text file failed')
    exit(0)

if __name__ == "__main__":
    main()
