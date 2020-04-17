# Installing a fresh virtual environment for envirocar-py
virtualenv allows you to manage separate package installation for different projects. They essentially allow you to create a “virtual” isolated Python installation and install packages into that virtual installation. When you switch projects, you can simply create a new virtual environment and not have to worry about breaking the packages installed in the other environments. It is always recommended to use a virtual environment while developing Python applications ([link](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)).


## Creating the environment
Create some folder where you want to make your project and virtualenv available. 
```
mkdir ./dev/envirocar/some_dir && cd ./dev/envirocar/some_dir
```

If virtualenv hasn't been installed before, you can install it via pip.
```
sudo pip3 install virtualenv
```

Create a new environment in the venv folder (second argument)
```
python3 -m venv venv
```

Activate the created virtual environment
```
source venv/bin/activate
```

You can now check whether the environment has been activated by checking the destination of `python`. It should link to the created folder.
```
which python
```

## Install the required dependencies
Install the python envoricar python package. This package will also install all necessary libraries such as Pandas or GeoPandas. 
```
pip install envirocar-py --upgrade
```

## Installing Virtualenv as a Python Kernel (for Jupyter)
Finally, add the virtualenv as a Jupyter kernel. In this way you can use the installed dependencies in venv in your Jupyter sessions.
```
ipython kernel install --user --name=envirocar
```

To ensure that the package is installed properly, the following command can be used.
```
jupyter kernelspec list
```

Now you can start a Jupyter notebook and select the created kernel `envirocar` as a new kernel for your Jupyter notebooks. To manage this, open your Jupyter notebook and click on Kernel -> Change Kernel -> envirocar.

Important: If you install any further dependencies, please make sure that you active this virtual environment before installing packages.
