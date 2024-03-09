# Application of predictive machine learning in pen & paper RPG game design

This project is the core of the master's thesis titled:
**Application of predictive machine learning in pen & paper RPG game design**.

It is a continuation of Engineering thesis:
*Application of machine learning to support pen & paper RPG game design*
([github](https://github.com/Paulina100/ML-for-TTRPG-game-design)).

### Author:

* Jolanta Śliwa

## Table of contents

* [Project Structure](#project-structure)
* [Technologies](#technologies)
* [Testing](#testing)

## Project Structure

* `notebooks`: Jupyter Notebooks
* `test`: tests
* `training`: scripts for creating datasets and training model

## Technologies

* Python 3.10
* Jupyter Notebook

## Testing

This project for testing uses the `pytest` framework.

To run all tests from `tests` directory, use:

```shell
make tests
```

To run tests from a specific file in `tests` directory, type:

```shell
make test FILE={filename.py}
