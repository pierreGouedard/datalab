# Charlin ML repo
[![Python][python-image]][python-url]
[![Build Status][circleci-image]][circleci-url]
[![Black][black-image]][black-url]
[![codecov][codecov-image]][codecov-url]

### Onboarding

Create virtual env and install janeml module

```
cd ~/ml-analysis
conda create -f conda_environment.yml
pip install --no-cache-dir -e .
```

Explore kedro client

```
kedro --help
```

Be carefull on local machine please always use --debug True. 


### Running pipelines locally 

```
kedro run -p <pipeline_name> --debug
```

### Running tests locally
The tests require the setup of local Mongo and Timescale databases to ingest standardized testing data.
These databases require a local install of the docker engine. Once docker is installed, `cd` into the `tests` folder and run:
    
    docker-compose up
This will use the `tests/docker-compose.yml` file to setup 2 containers, one for each database. 

The tests can the be run from the root directory with the `pytest` command. 
Pytest will use the additional configuration outlined in the `setup.cfg` file (`[tool:pytest]` section).

To stop the docker containers run:

    docker-compose down
    

### Code coverage
When running the tests the code coverage will be calculated automatically and `coverage.xml` report will be generated.
The code coverage reports can be checked at [codecov][codecov-url] and will be auto uploaded from CircleCI when code is pushed on the master branch.

To manually push a local codecov report run:

    bash <(curl -s https://codecov.io/bash)

### Updating the documentation
The technical documentation can be auto-generated from the docstrings in the code using the [Sphinx][sphinx-url] package.

To get started, make sure the packages specified in `requirements_dev.txt` are installed.
Then `cd` into the `docs` folder and run:

    make clean
To remove old build files. Then:

    make html
To create the new html files.
You can then inspect the docs with

    open build/html/index.html

<!-- Variables -->
[python-image]:https://img.shields.io/badge/python-3.7-blue.svg
[python-url]:https://www.python.org/downloads/release/python-375/
[circleci-image]:https://img.shields.io/circleci/build/github/daubechies/charlin-ml/master?token=0051d08104943b58a44bc30c7724683f87859068
[circleci-url]:https://circleci.com/gh/daubechies/charlin-ml/tree/master
[black-image]:https://img.shields.io/badge/code%20style-black-000000.svg
[black-url]:https://github.com/ambv/black
[codecov-image]:https://codecov.io/gh/daubechies/charlin-ml/branch/master/graph/badge.svg?token=Wi7x6Ae0Zf
[codecov-url]:https://codecov.io/gh/daubechies/charlin-ml
[sphinx-url]:https://www.sphinx-doc.org/en/master/index.html
