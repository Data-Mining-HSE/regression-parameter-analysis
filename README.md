# Regression Parameter Analysis

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
```
