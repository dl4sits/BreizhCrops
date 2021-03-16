## Tests



this folder contains tests for

* `package.py` lightweight tests executed on the github server at each commit 
* `data.py` heavy tests to check the download and execution of the dataset
* `models.py` heavy tests to check the model performance (to be implemented)

run tests via
```
export TESTS_DATA_ROOT="/ssd/breizhcrops"
pytest tests/package.py
```
you can select the folder to store the data via the environment variable `TESTS_DATA_ROOT`. if not provided, 
test default to `/tmp`
