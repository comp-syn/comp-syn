#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from setuptools import setup

DESCRIPTION = ''
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

version = "0.2.2"
setup(name='compsyn',
      version=version,
      description='python package to explore the color of language',
      author='Bhargav Srinivasa Desikan',
      author_email='bhargavvader@gmail.com',
      url='https://github.com/comp-syn/comp-syn',
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
          'black',
          'numpy',
          'scipy',
          'pillow',
          'matplotlib',
          'memory_profiler',
          'numba',
          'seaborn',
          'scikit-learn',
          'google_images_download', #for downloading images
          'google-cloud-vision',
          'textblob',
          'nltk',
          'random_word',
          'selenium',  
          'bs4'
      ],
      keywords=[
          'Image Analysis',
          'Computational Synaesthesia',
          'Color Science',
          'Computational Social Science'
          'Cognitive Science'
      ],
      long_description=LONG_DESCRIPTION)
