from pip._internal.req import parse_requirements
from setuptools import setup, find_packages

# loading requirements
requirements = list(parse_requirements('requirements.txt', session='hack'))
requirements = [r.name for r in requirements]

setup(
    name="enpyrocar",
    packages=find_packages(exclude=["tests", "tests.*"]),
    version="0.1",
    license="MIT",
    url="https://github.com/enviroCar/enpyrocar",
    keywords=["enviroCar", "trajectory"],
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