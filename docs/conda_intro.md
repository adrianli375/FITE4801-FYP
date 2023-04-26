# Introduction to Virtual Environment

A virtual environment (often referred to as a "virtualenv") is a tool in Python that allows you to create an isolated environment for a Python project. This environment contains its own installation of Python and its own set of packages, which can be different from the packages installed on the system's global Python installation. Virtualenvs are useful when you want to work on multiple Python projects with different dependencies or when you want to avoid conflicts between packages. By creating a virtualenv, you can install the required packages for each project separately and avoid potential compatibility issues.

To avoid compatibility issues as mentioned above, we need to properly set up a virtual environment. This is achieved by installing Conda, which is a popular package manager and environment management system for Python programming language. 

Before setting up a Virtual Environment, we need to install the dependencies first. 

# Install Conda

You can download the Miniconda Python distribution which comes with a bunch of useful packages, such as NumPy, scikit-learn and Jupyter notebook. You can download Miniconda [here](https://docs.conda.io/en/latest/miniconda.html) which is suitable for your operating system. It make take around 15-30 minutes for installation. 

After finishing installation, you can check the following things to ensure conda is installed. 

1. Reopen your terminal and you can see the prompt looks like the following: 
```
(base) username@MACHINE-NAME:~$
```
2. To confirm that conda is working, you can check the version of conda using the following command: 
```
conda --version
```
For example, it will return something like
```
conda 23.1.0
```

# Check Python version

Type the following command in your terminal to check that Python is installed in your local machine: 
```
python --version
```
For example, it will return something like 
```
Python 3.9.16
```
Ensure that the version of Python is 3.X.X or above. 

# Install Python packages
`conda` installs Python packages from different online repositories. [`conda-forge`](https://conda-forge.org/) is a community-driven effort which provides up to date packages in Python. Add the `conda-forge` channel by typing the following command in your terminal: 
```
conda config --add channels conda-forge
```

# Setting up a Virtual Environment

1. Check the list of environments by running
```
conda env list
```
For example, the output of my machine is 
```
# conda environments:
#
base                  *  /Users/adrianli375/miniconda3
course1                  /Users/adrianli375/miniconda3/envs/course1
```
which means that I have two virtual environments: `base` and `course1`. The asterisk (*) indicates that the current environment that I am in is the `base` (default) environment. 

2. Download [`fypenv.yml`](https://github.com/adrianli375/Test-Project/blob/main/docs/fypenv.yml) and move the file to your working directory. 

3. Create an environment by running the command
```
conda env create -f fypenv.yml
```
which tells `conda` to download the packages and dependencies needed and put them in a virtual environment named `fyp`. The environment should be installed successfully, as you may check `conda env list` again that the environment `fyp` exists in the output. 

4. Activate our newly created environment by 
```
conda activate fyp
```
After successfully activated the environment, `(fyp)` should show up in the command line of the terminal. 

5. (Optional) To deactivate the environment, run the following command in the terminal: 
```
conda deactivate
```
