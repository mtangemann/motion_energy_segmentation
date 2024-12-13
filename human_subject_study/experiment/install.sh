#!/usr/bin/env bash

set -ex

python3.11 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
