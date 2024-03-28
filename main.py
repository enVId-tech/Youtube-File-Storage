from decode import decode_video
from encode import encode_file


def main():
    try:
        print("Running encode...")
        encode_file()

        print("\n\n\n----------------------------------------------\n\n\n")

        print("Running decode...")
        decode_video()

        input_checksum = 0 # compute_checksum(f'./input_files/{INPUT_PATH}')
        output_checksum = 0 # compute_checksum(f'./output_files/{OUTPUT_FILE}')

        if input_checksum and output_checksum:
            if input_checksum == output_checksum:
                print("\n\nChecksums match!")
            else:
                print("Checksums do not match!")
        else:
            print("Checksums could not be computed!")

        print(f"\nInput checksum: {input_checksum}")
        print(f"Output checksum: {output_checksum}")
        exit(0)
    except Exception as e:
        print(f"Error in main(): {e}")
        exit(1)

if __name__ == "__main__":
    main()