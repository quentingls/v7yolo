import os
from setuptools import setup, find_packages

setup(
    name='v7yolo',
    version='20210221',
    description='Utility package to train yolo models on v7 labelled data',
    url='https://github.com/quentingls/v7yolo',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'v7yolo=v7yolo.cli:v7yolo'
        ]
    },
    install_requires=list(open('requirements.txt').read().strip().split('\n'))
)
