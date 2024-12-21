# -*- coding: utf-8 -*-
"""
Setup information for the project.

Code initally generated for the BDA challenge in SS 2022.
"""
from setuptools import find_packages, setup

setup(
    name="bda case challenge",
    version="0.1.0",
    description="Project to predict the load profiles of households",
    author="Yacin Boualili",
    author_email="yacin.boualili@student.kit.edu",
    packages=find_packages(exclude=("tests", "docs")),
)
