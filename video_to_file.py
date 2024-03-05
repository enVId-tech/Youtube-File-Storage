from filehandling.to_file_funcs import video_convert_to_file

DEVICE = 'cpu'

def main():
    # Convert the video to a text file
    vidToTxt = video_convert_to_file('./output_files/mainvideo.mp4', DEVICE)
    if vidToTxt:
        print('Video converted to text file successfully')
    else:
        print('Video conversion to text file failed')
    exit(0)

if __name__ == "__main__":
    main()
