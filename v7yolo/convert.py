import os
import json
import requests
import collections
import glob
import typing

_BoundingBox = collections.namedtuple('_BoundingBox', ('x', 'y', 'w', 'h'))


def _download_image(img_config: dict, target_dir: str) -> str:
    """
    Downloads the image locate at the `url` location
    :param img_config: file configurations dict
    :return: None
    """
    url = img_config['url']
    name = img_config['filename']
    img = requests.get(url).content
    path = os.path.join(target_dir, name)
    print(f'downloading image at {url} to {path}')
    with open(path, 'wb') as f:
        f.write(img)

    return path


def _map_annotation(configs: typing.List[dict]) -> typing.Dict[str, int]:
    """
    Maps annotation names to their alphabetical indices
    :param configs: Files configurations containing named annotations
    :return: A dict mapping from name to index
    """
    # building label_map
    labels = []
    for config in configs:
        for annotation in config['annotations']:
            labels.append(annotation['name'])

    # make mapping deterministic
    labels = list(set(labels))
    labels.sort()
    return {labels[i]: i for i in range(len(labels))}


def _build_bounding_box(width: int, height: int, polygon: dict) -> _BoundingBox:
    """
    Builds
    :param width: The image width
    :param height: The image height
    :param polygon: The polygon points
    :return: A bounding box with normalized coordinates
    """
    points = polygon['path']
    # getting corners
    x_min, x_max, y_min, y_max = width, 0, height, 0
    for point in points:
        x, y = point['x'], point['y']
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    # calc center
    c_x = (x_min + x_max) / 2
    c_y = (y_min + y_max) / 2

    # cal w & h
    box_w = x_max - x_min
    box_h = y_max - y_min

    # normalizing coord
    c_x = round(c_x / width, 2)
    c_y = round(c_y / height, 2)
    box_w = round(box_w / width, 2)
    box_h = round(box_h / height, 2)

    return _BoundingBox(x=c_x, y=c_y, w=box_w, h=box_h)


def _v7_to_yolo_annotation(config: dict, target_dir: str, label_map: dict):
    """
    Converts a V7 polygon annotation to a Yolo bounding box annotation
    :param config: the v7 image config
    :param target_dir: path to the directory to save the annotations and downloaded image
    :param label_map: mapping from v7 generated name to integer index
    :return: None
    """
    img_config = config['image']

    width = img_config['width']
    height = img_config['height']
    annotations = config['annotations']

    file_name = f'{os.path.splitext(img_config["filename"])[0]}.txt'
    file_path = os.path.join(target_dir, file_name)
    print(f'writing annotation to {file_path}')

    with open(file_path, 'w') as f:
        for annotation in annotations:
            name = annotation['name']
            idx = label_map[name]
            bb = _build_bounding_box(width, height, annotation['polygon'])
            line = f'{idx} {bb.x} {bb.y} {bb.w} {bb.h}'
            f.write(f"{line}\n")


def v7_to_yolo(input_dir: str, target_dir: str, download=True, label_map_path: str = None):
    """

    :param input_dir: path to the directory containing the v7 annotation images
    :param target_dir: path to the directory to save the annotations and downloaded image
    :param label_map_path: path to file mapping from v7 generated name to integer index
    :param download: If set to True it will try to download the file specified at the url key
    :return: None
    """
    v7_file_paths = glob.glob(os.path.join(input_dir, '*.json'))
    # Read all configs
    configs = []
    for path in v7_file_paths:
        print(f'reading config {path}')
        with open(path) as f:
            configs.append(json.load(f))

    if download:
        print('downloading images...')
        for config in configs:
            _download_image(config['image'], target_dir)

    # building label_map
    if not label_map_path:
        print('mapping annotation class to alphabetical index...')
        label_map = _map_annotation(configs)
        with open(os.path.join(target_dir, 'map.json'), 'w') as f:
            json.dump(label_map, f)
    else:
        with open(label_map_path, 'r') as f:
            label_map = json.loads(f)

    # building yolo annotations
    print('building yolo annotation from v7 configs')
    for config in configs:
        _v7_to_yolo_annotation(config, target_dir, label_map)

    print('all done...')
