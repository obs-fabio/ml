#!/bin/bash
python3 setup.py bdist_wheel
pip install dist/ml*.whl --force-reinstall
