# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import os
import re

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

install_requires = [
    'numpy >= 1.16',
    'opencv-python',
    'pyudev'
]

dependency_links = [
]


def read(fname):
    return open(os.path.join(BASE_DIR, fname)).read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='digit_interface',
      version=find_version("digit_interface/__init__.py"),
      description='Interface for the DIGIT tactile sensor.',
      url='https://github.com/fairinternal/digit-interface',
      author='Mike Lambeta, Roberto Calandra',
      author_email='lambetam@fb.com, rcalandra@fb.com',
      keywords=['science'],
      long_description=read('README.md'),
      license='LICENSE',
      packages=find_packages(),
      install_requires=install_requires,
      dependency_links=dependency_links,
      zip_safe=True,
      classifiers=[
          'Programming Language :: Python :: 3',
          "Operating System :: POSIX :: Linux",
      ],
      )
