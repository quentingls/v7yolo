import os
import json
import requests
import collections
import glob
import typing
import yaml
from shutil import copyfile


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


def _labels(configs: typing.List[dict]) -> typing.List[str]:
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
    return labels


def _training_config(target_dir: str, labels: typing.List[str]) -> typing.Dict:
    return {
        'train': os.path.join(target_dir, 'train'),
        'val': os.path.join(target_dir, 'val'),
        'nc': len(labels),
        'names': labels
    }


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


def v7_to_yolo(input_dir: str, target_dir: str, download=True, split: float = 0.3):
    """

    :param input_dir: path to the directory containing the v7 annotation images
    :param target_dir: path to the directory to save the annotations and downloaded image
    :param download: If set to True it will try to download the file specified at the url key
    :param split: Train / Test split ratio
    :return: None
    """
    v7_file_paths = glob.glob(os.path.join(input_dir, '*.json'))
    # Read all configs
    configs = []
    for path in v7_file_paths:
        print(f'reading config {path}')
        with open(path) as f:
            configs.append(json.load(f))

    # building label_map
    print('building training configuration...')
    labels = _labels(configs)
    label_map = {labels[i]: i for i in range(len(labels))}
    training_config = _training_config(target_dir, labels)
    with open(os.path.join(target_dir, 'config.yaml'), 'w') as f:
        yaml.dump(training_config, f)

    train_configs, valid_configs = configs[int(len(configs) * split):], configs[0: int(len(configs) * split)]
    for config in train_configs:
        os.makedirs(training_config['train'], exist_ok=True)
        config['target_dir'] = training_config['train']
    for config in valid_configs:
        os.makedirs(training_config['val'], exist_ok=True)
        config['target_dir'] = training_config['val']

    for configs in (train_configs, valid_configs):
        # building yolo annotations
        print('building yolo annotation from v7 configs')
        for config in configs:
            _v7_to_yolo_annotation(config, config['target_dir'], label_map)

        if download:
            print('downloading images...')
            print(f'images will be split between train / validation set ({1 - split} / {split})')
            for config in configs:
                _download_image(config['image'], config['target_dir'])
        else:
            print(f'splitting images between train / validation set ({1 - split} / {split})')
            for config in configs:
                path = os.path.join(input_dir, config['image']['filename'])
                name = os.path.basename(path)
                target_path = os.path.join(config['target_dir'], name)
                copyfile(path, target_path)

    print('all done...')


if __name__ == '__main__':
    v7_to_yolo(
        input_dir='/Users/quentin/projects/v7yolotest/inputs',
        target_dir='/Users/quentin/projects/v7yolotest',
        download=False
    )
