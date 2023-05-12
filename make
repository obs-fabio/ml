#!/bin/bash

if [ $# -eq 0 ]; then
  python3 setup.py bdist_wheel
  exit 0
fi

case $1 in
    install)
        python3 setup.py bdist_wheel
        pip install dist/ml*.whl --force-reinstall
        ;;
    list)
        echo $0 "[install list]"
        ;;
    *)
        echo "invalid argument, for details:" $0 "list"
    ;;
esac


