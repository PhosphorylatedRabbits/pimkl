#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = [
        line.strip() for line in requirements_file if not ('mimkl' in line)
    ]

setup_requirements = []

test_requirements = []

setup(
    author="Joris Cadow and Matteo Manica",
    author_email='joriscadow@gmail.com, drugilsberg@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description=
    "pathway induced multiple kernel learning for computational biology",
    entry_points={
        'console_scripts':
            [
                'pimkl=pimkl.cli:main',
                'pimkl-preprocess=pimkl.cli.preprocess:main',
                'pimkl-analyse=pimkl.cli.analyse:main'
            ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pimkl',
    name='pimkl',
    packages=find_packages(include=['pimkl', 'pimkl.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/PhosphorylatedRabbits/pimkl',
    version='0.1.1',
    zip_safe=False,
)
