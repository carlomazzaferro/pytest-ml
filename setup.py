#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()


setup(
    name='pytest-ml',
    version='0.1.0',
    author='Carlo Mazzaferro',
    author_email='carlo.mazzaferro@gmail.com',
    maintainer='Carlo Mazzaferro',
    maintainer_email='carlo.mazzaferro@gmail.com',
    license='MIT',
    url='https://github.com/carlomazzaferro/pytest-ml',
    description='Test your machine learning!',
    long_description=read('README.rst'),
    py_modules=['pytest_ml'],
    install_requires=['pytest>=3.5.0', 'test-ml'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: Pytest',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points={
        'pytest11': [
            'ml = pytest_ml.plugin',
        ],
    },
)
