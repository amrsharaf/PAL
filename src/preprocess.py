import argparse

# Parse command line flags
def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', help='file name', required=True)
    ap.add_argument('--output', '-o', help='output file', required=True)
    return ap.parse_args()

def cleanup_line(line):
    splits = line.strip().split()
    if len(splits) == 0:
        return line.strip()
    else:
        return ' '.join([splits[0], splits[-1]])

def main():
    args = parse_arguments()
    input_file = args.input
    output_file = args.output
    print('reading data from: ', input_file)
    print('writing data to: ', output_file)
    with open(output_file, 'w', encoding='utf-8') as output_writer:
        print('successfully opened output file...')
        with open(input_file, 'r', encoding='latin-1') as input_reader:
            lines = input_reader.readlines()
            for line in lines:
                new_line = cleanup_line(line)
                output_writer.write(new_line)
                output_writer.write('\n')
    print('done with pre-processing...')

if __name__ == '__main__':
    main()
