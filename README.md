# Machine Learning analysis of Abalone dataset

## Get a data
```bash
mkdir -p data/abalone
pushd data/abalone
wget https://archive.ics.uci.edu/static/public/1/abalone.zip
unzip abalone.zip
popd
```

## Prepare environment

| Use python version 3.9+

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Regression Parameter Analysis

See `regression_parameter_analysis.ipynb`

## Gradient Descent Analysis

See `gradient_descent.ipynb`
