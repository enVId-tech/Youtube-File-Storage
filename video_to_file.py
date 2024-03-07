from filehandling.to_file_funcs import video_convert_to_file

DEVICE = 'cpu'

INPUT_FILE = 'mainvideo.mp4'


def main():
    # Convert the video to a text file
    vidToTxt = video_convert_to_file(f'./output_files/{INPUT_FILE}', DEVICE)
    if vidToTxt:
        print(
            f'{INPUT_FILE.split(".")[-1].upper()} video converted to file successfully.'
        )
        exit(0)
    else:
        print(
            f'{INPUT_FILE.split(".")[-1].upper()} video conversion to file failed.'
        )
        exit(1)


if __name__ == "__main__":
    main()
