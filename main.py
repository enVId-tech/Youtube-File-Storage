from proj_funcs.decode import decode_video
from proj_funcs.encode import encode_file
from proj_funcs.misc import compute_checksum
from constants import INPUT_PATH, OUTPUT_FILE

def main():
    try:
        print("Running encode...")
        enc = encode_file()

        print("\n\n----------------------------------------------\n\n")

        if not enc:
            print("Encode failed!")
            exit(1)

        exit(0)

        print("Running decode...")
        decode_video()

        if not enc:
            print("Decode failed!")
            exit(1)

        print("\n\n----------------------------------------------\n\n")

        input_checksum = compute_checksum(f'./input_files/{INPUT_PATH}')
        output_checksum = compute_checksum(f'./output_files/{OUTPUT_FILE}')

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