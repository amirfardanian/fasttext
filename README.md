# README

## About

`capture_model` is a library to train, validate and use text classifiers for categorizing and classifying 'capture lines' (rows from receipts).
It uses Facebook's Fasttext under the hood, and provides functionality to define and train pre-processors and models.

## Installation

To install the library, run 

```bash
pip install git+ssh://git@gitlab.data4insight.com/ds/capture_model.git
```

or add 

`git+ssh://git@gitlab.data4insight.com/ds/capture_model.git` to your _requirements.txt_ file.

## Usage

#### Categorizing and classifying receipt a list of receipt lines.

The module for direct usage for classification is `capture_model.scoring`.
This module contains 4 main functions:

- `predict_line(line: str)`: preprocesses and predict the category_id of a string.
It returns a dictionary containing the predicted _category_id_, the prediction _score_,
and the text _line_. 
- `predict_lines(lines:  List[str])`: runs `predict_line()` on a list of strings.
- `classify_line(predicted_line: str)`: takes a predicted_line dictionary and classifies it.
Returns a dictionary containing the _CLASS_, the class _VALUE_, and the line _TEXT_. 
_ `classify_lines(predicted_lines: List[str])`: runs `classify_line()` on a list of predicted lines.

Example usage:

```python
import capture_model.scoring as sf

lines = ['cvr eee 333333', 'TOTAL 199,95', 'kanelsnegl 13.00']

predicted_lines = sf.predict_lines(lines)
classified_lines = sf.classify_lines(predicted_lines)

for scored_line in classified_lines:
	print(scored_line)
```

#### Defining a PreProcessor

A serializable PreProcessor object can be defined using the PreProcessor class
in capture_model. 

Example usage:

```python
from capture_model.modelling import PreProcessor
from capture_model.modelling.transformations import StatelessTransformation
from capture_model.modelling import string_functions as strings

preprocessor = PreProcessor('my_preprocessor')

preprocessor.add_transformation(StatelessTransformation(function=strings.lower))  # Convert all to lowercase
preprocessor.add_transformation(StatelessTransformation(function=strings.replace, regex_pattern=r'[^a-zæøåäö ]', replacement=''))  # Remove numerical and non-nordic characters
preprocessor.add_transformation(StatelessTransformation(function=strings.replace, regex_pattern=r'(\s)+', replacement=' '))  # Replace whitespace with single space
preprocessor.add_transformation(StatelessTransformation(function=strings.ljust, width=1, fillchar=' '))  # Ensure that lines are not empty.

preprocessor.to_json('my_preprocessor.json')
```

#### Training a text classifier
... Coming soon ...


#### Validating a text classifier's performance
... Coming soon ...

## Running tests

```
cd tests
PYTHONPATH=../ python run_tests.py
```
