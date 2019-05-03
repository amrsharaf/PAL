from argparse import ArgumentParser


def is_english_line(line):
    return line.startswith('en:')


def is_german_line(line):
    return line.startswith('de:')


def is_spanish_line(line):
    return line.startswith('es')


def is_dutch_line(line):
    return line.startswith('nl')


def get_filter(language):
    if language == 'en':
        return is_english_line
    elif language == 'de':
        return is_german_line
    elif language == 'es':
        return is_spanish_line
    elif language == 'nl':
        return is_dutch_line
    else:
        assert False


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--language', required=True, type=str, help='Lanuage')
    parser.add_argument('--input_path', required=True, type=str, help='Input path')
    parser.add_argument('--output_path', required=True, type=str, help='Output path')
    return parser.parse_args()


def main():
    args = parse_arguments()
    language =  args.language
    input_file_path = args.input_path
    output_file_path = args.output_path
    print('Reading embeddings from: ', input_file_path)
    with open(input_file_path, 'r', encoding='utf-8') as file_reader:
        print('opened file!')
        # Read all input lines
        lines = file_reader.readlines()
        # Filter results to include only the language words!
        is_language_filter = get_filter(language)
        english_lines = filter(is_language_filter, lines)
        # Write the language lines to an output file
        with open(output_file_path, 'w', encoding='utf-8') as output_writer:
            output_writer.writelines(english_lines)
        print('done writing file to: ', output_file_path)


if __name__ == '__main__':
    main()
