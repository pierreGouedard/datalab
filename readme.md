# Kedro orchestrated data re repo
[![Python][python-image]][python-url]
[![Black][black-image]][black-url]
[![codecov][codecov-image]][codecov-url]

### Onboarding

Create virtual env and install janeml module

```
cd ~/datalab
conda create -f cvenv_dev.yml
pip install --no-cache-dir -e .
```

Explore kedro client

```
kedro --help
```

Be carefull on local machine please always use --debug True. 


### Running pipelines 

```
kedro run -p <pipeline_name> 
```

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
