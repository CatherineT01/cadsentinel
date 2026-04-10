#setup.py

from setuptools import setup, find_packages

setup(
    name="cadsentinel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "psycopg2-binary>=2.9.9",
        "openai>=1.30.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0"],
    },
)
