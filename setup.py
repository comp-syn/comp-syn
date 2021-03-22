#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import toml
from pathlib import Path
from setuptools import setup
import subprocess

with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()


PYPROJECT = toml.loads(Path(__file__).parent.joinpath("pyproject.toml").read_text())

def install_requires() -> List[str]:
    """ Populate install_requires from requirements.txt """
    requirements_txt_proc = subprocess.run(["poetry", "export", "-f", "requirements.txt"], capture_output=True, check=True)
    requirements_text = requirements_txt_proc.stdout.decode("utf-8")

    return [ requirement for requirement in requirements_txt.split("\n") ]


setup(
    name=PYPROJECT["tool"]["poetry"]["name"],
    version=PYPROJECT["tool"]["poetry"]["version"],
    description=PYPROJECT["tool"]["poetry"]["description"],
    author="Compsyn Group",
    author_email="group@comp-syn.com",
    url="https://github.com/comp-syn/comp-syn",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    packages=["compsyn"],
    install_requires=install_requires(),
    keywords=[
        "Image Analysis",
        "Computational Synaesthesia",
        "Color Science",
        "Computational Social Science",
        "Cognitive Science",
    ],
    long_description=LONG_DESCRIPTION,
)
