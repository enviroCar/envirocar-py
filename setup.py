from pip._internal.req import parse_requirements
from setuptools import setup, find_packages

requirements = parse_requirements('requirements.txt', session='hack')

setup(
    name="enpyrocar",
    version="0.1",
    packages=find_packages(exclude=["tests", "tests.*"]),
#    install_requires=requirements,
    test_suite="tests"
)