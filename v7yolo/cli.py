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
    parser.add_argument('--map', action='store', required=False, default=None,
                        help='Path the map of labels to index. If not set the map will be generated.')
    return parser


def v7yolo():
    args = _build_arg_parser().parse_args(sys.argv[1:])
    v7_to_yolo(args.input, args.dest, args.download, args.map)
