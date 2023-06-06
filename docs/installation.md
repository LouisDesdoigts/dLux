# Installation

∂Lux is hosted on PyPI, so simply pip install!

```
pip install dLux
```

You can also build from source. To do so, clone the git repo, enter the directory, and run

```
pip install .
```

We encourage the creation of a virtual environment to run ∂Lux to prevent software conflicts as we keep the software up to date with the latest version of the core packages.

## Windows/Google Colab Quickstart
`jaxlib` is currently not supported by the jax team on windows, however there are two work-arounds!

Firstly [here](https://github.com/cloudhan/jax-windows-builder) is some community built software to install jax on windows! We do not use this ourselves so have limited knowledge, but some users seems to have got everyting working fine!

Secondly users can also run our software on [Google Colab](https://research.google.com/colaboratory/). If you want to instal from source in colab, run this at the start of your notebook!

```
!git clone https://github.com/LouisDesdoigts/dLux.git # Download latest version
!cd dLux; pip install . -q # Navigate to ∂Lux and install from source
```

From here everything should work! You can also run the code on GPU to take full advantage of Jax, simply by switch to a GPU runtime environment, no extra steps necessary!