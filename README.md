# envirocar-py

The enviroCar Python package allows users to query and download trajectory data from the enviroCar API. The data will be stored in a flat GeoDataFrame from GeoPandas. The resulting dataframe consists of all measurements from the requested tracks including attached sensorinformation from the cars and further meta infromation about the track. 

The package currently only supports querying track data. It is intended to further expand these functionalities with additional analytics in the future.

## Installation

The package requires a Python version >= 3.6. The package is available on the PyPI package manager and can be installed with the following command:

```
pip install envirocar-py --upgrade
```

To install envirocar-py in develop mode, use the following:

```
python setup.py develop
```

## Examples
Jupyter notebooks on how to use this package to request data from enviroCar API can be found in the examples folder:
 * Download data and visualize with pydeck ([here](https://github.com/enviroCar/envirocar-py/blob/master/examples/api_request_deckgl.ipynb))

## License ##
    MIT License

    Copyright (c) 2020 The enviroCar Project

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
