# JupyterLab

[JupyterLab](https://jupyter.org/) is an interactive development environment (IDE) for working with Jupyter notebooks, code, and data. It provides a modern and flexible user interface that allows you to organize your work into notebooks, terminal windows, text editors, and more, all within a single web-based application. With JupyterLab, you can write and execute code, create visualizations, and explore data using a wide range of programming languages, including Python, R, and Julia. JupyterLab also supports the installation of third-party extensions, which can add additional functionality to the IDE, such as interactive widgets, enhanced text editors, and more. 

To install JupyterLab on your device, run the command
```
conda install jupyterlab jupyterlab-spellchecker nb_conda_kernels
```

Please ensure that `conda` is installed before running this command. See [`conda_intro.md`](conda_intro.md) for details. 

Note that you only need to install this in the `base` virtual environment. There is no need to install JupyterLab in other virtual environments. 

# Open JupyterLab to edit code

Start JupyterLab from your local machine by typing the command
```
jupyter lab
```
in the terminal from your `base` environment (NOT from any other virtual environments!)

Jupyter Lab will be opened in your default browser. Under the tab `Kernel`, choose the appropriate kernel as the preferred one. You are all set! 
