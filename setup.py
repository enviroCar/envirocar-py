import io

from pip._internal.req import parse_requirements
from setuptools import setup, find_packages

# loading requirements
requirements = list(parse_requirements('requirements.txt', session='hack'))
requirements = [r.name for r in requirements]


def parse_long_description():
    return io.open('README.md', encoding="utf-8").read()

setup(
    name="envirocar-py",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "": ["*.txt"]
    },
    include_package_data=True,
    version="0.0.1",
    description="Python Utilities for enviroCar",
    long_description=parse_long_description(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/enviroCar/envirocar-py",
    keywords=["enviroCar", "trajectory", "xFCD"],
    install_requires=requirements,
    test_suite="tests",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)