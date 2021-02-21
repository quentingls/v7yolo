import argparse
import sys

from v7yolo import v7_to_yolo


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store', required=True,
                        help='The directory containing the v7 annotations.')
    parser.add_argument('--dest', action='store', required=True,
                        help='The folder destination for the annotations and images')
    parser.add_argument('--download', action='store_true', required=False, default=False,
                        help='Flag indicating if to download the images from url')
    parser.add_argument('--split', action='store', required=False, default=0.3,
                        help='Train validation test split (equals to the percentage reserved for validation)')
    return parser


def v7yolo():
    args = _build_arg_parser().parse_args(sys.argv[1:])
    v7_to_yolo(args.input, args.dest, args.download, args.split)
