#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
cd ..

wget https://storage.googleapis.com/breizhcrops/data.zip
unzip data.zip
rm data.zip # cleanup