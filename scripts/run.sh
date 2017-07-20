#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR

./calculate_multivar.sh > ../data/data_multivar.csv
./calculate_mixture.sh > ../data/data_mixture.csv

./plot.sh

cd -
