#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from setuptools import setup

DESCRIPTION = ''
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

version = "0.1.0"
setup(name='compsyn',
      version=version,
      description='Python package to explore the color of language',
      author='Bhargav Srinivasa Desikan',
      author_email='bhargavvader@gmail.com',
      url='https://github.com/bakerwho/comp-syn',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      packages=['compsyn'],
      install_requires=[
          'numpy',
          'scipy',
          'pillow',
          'matplotlib',
      ],
      keywords=[
          'Image Analysis',
          'Computational Syn',
          'Cognitive Science'
      ],
      long_description=LONG_DESCRIPTION)
