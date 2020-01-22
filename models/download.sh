#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

wget https://storage.googleapis.com/breizhcrops/models.zip
unzip models.zip
rm models.zip # cleanup
