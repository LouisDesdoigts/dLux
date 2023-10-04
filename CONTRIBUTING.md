# Contributing Guide

dLux is an open-source framework and as such is very welcoming to contributions via pull requests! If you would like to contribute but are unfamiliar with any of this process, as I imagine many coming from a science background will be, please feel free to reach out to me at my [email](mailto:louis.desdoigts@sydney.edu.au) or on [twitter](https://twitter.com/gradientrider) and I will be happy to help you through the process!

---

## Getting Started

Firstly, you will need to fork the repository to your own github account. This will allow you to make changes to the code and then submit a pull request to the main repository. To do this, click the fork button in the top right of the repository page. This will create a copy of the repository in your own account that you can make changes to and then request to merge with the main repository.

Next, you will need to clone the repository to your local machine. To do this, open a terminal and navigate to the directory you would like to clone the repository to. Then run the following command:

```bash
https://github.com/your-username-here/dLux.git 
cd dLux
pip install '.[dev]'
```

Then you will need to install the pre-commit hooks. This will ensure that the code is formatted correctly and that the unit tests pass before you can commit your changes. To do this, run the following command:

```bash
pre-commit install
```

This will ensure that any changes you make will adhere to the code style and formatting guidelines of the rest of the package!

---

## Making Changes

Next you can start to make any changes you desire!

**Unit Tests**

It is important that any changes you make are tested to ensure that they work as intended and do not break any existing functionality. If you are creating _new_ functionality you will need to create some new unit tests, otherwise you should be able to modify the existing tests.

To ensure that everything is working as expected, you can run the unit tests by running the following command:

```bash
pytest tests/*
```

This will run all the scripts labelled with `test_` in the `tests` directory. If you would like to run a specific test, you can run the following command:

```bash
pytest tests/test_file.py
```

Note that just because the tests pass on your local machine, that does not mean that it will necessarily pass on all others! This can be due to a number of reasons such a different operating system, different python version, or different dependencies. This is why github actions are used to run the unit tests on a number of different operating systems and python versions. This should help ensure that the code works as expected on all platforms.

**Documentation**

Any changes you make should also be appropriately documented! For small API changes this shouldn't require any changes, however if you are adding new functionality you will need to add some documentation. This can be done by modifying the appropriates files in the `docs` directory.

To build the documentation locally and make sure everything is working correctly, you can run the following command:

```bash
mkdocs serve
```

This will build the documentation and serve it on a local server. You can then navigate to `localhost:8000` in your browser to view the documentation.

---

## Contributing the Changes

After these steps have been completed, you can commit your changes and push them to your forked repository. These changes should have its formatting and linting checked by the pre-commit hooks. If there are any issues, you will need to fix them before you can commit your changes. Once you have pushed your changes to your forked repository, you can submit a pull request to the main repository. This will allow the maintainers to review your changes and merge them into the main repository!
