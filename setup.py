#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2024/1/17
@author yrh

"""

from setuptools import setup


setup(
    name='clalign',
    version='1.0',
    packages=['clalign'],
    python_requires='>=3.12',
    install_requires=[
        'numpy>=2.0.2',
        'scipy>=1.14.1',
        'biopython>=1.84',
        'torch==2.4.1+cu124',
        'transformers>=4.41.2',
        'sentencepiece>=0.2.0',
        'protobuf>=5.28.3',
        'peft>=0.13.0',
        'numba>=0.60.0',
        'click>=8.1.7',
        'logzero>=1.7.0',
        'tqdm>=4.66.5'
    ],
    entry_points={
        'console_scripts': [
            'clalign=clalign.main:main'
        ]
    },

    author='yourh',
    author_email='yourh@nankai.com',
    description='',
    long_description='',
    long_description_content_type='text/markdown',
    url='https://github.com/yourh/CLAlign',
    classifiers=[]
)
