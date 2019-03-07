import argparse

# Parse command line flags
def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', '-f', help='file name', required=True)
    return ap.parse_args()

def cleanup_line(line):
    return None

def main():
    args = parse_arguments()
    file_name = args.file
    print('reading data from: ', file_name)
    with open(file_name, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            splits = line.strip().split()
            print(splits)

if __name__ == '__main__':
    main()
