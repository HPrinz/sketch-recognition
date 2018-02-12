#!/usr/bin/env bash

for f in `find . -name "*.png"`
do
    convert ${f} -resize 150x150 ${f}
done