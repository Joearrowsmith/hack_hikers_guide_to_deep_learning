# A Hack Hikers Guide To Deep Learning

## 1 - Virtual Environment

Creates a virutal environment called env
- python3 -m venv env 

Activate env windows: 
- env/Script/Activate
Activate env mac / linux: 
- source env/bin/Activate

## 2 - Create a folder for unit tests

mkdir code/tests

To run pytest, write:

python -m pytest

## 3 - Link to travis

touch .travis.yml

"""
language: python
python: 
 - "3.6"
install:
 - pip install -U pip
 - pip install -r "reqs.txt"
script:
 - pytest
"""










