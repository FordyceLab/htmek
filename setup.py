#!/usr/bin/env python

# For editable install

from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

def main():
    setup(
        name='htmek',
        description='Shared processing scripts for HT-MEK experiments that have been worked up by magnify.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/FordyceLab/htmek',
        install_requires=[
            'numpy',
            'pandas',
            'bokeh',
            'holoviews',
        ],
    )

if __name__ == "__main__":
    main()