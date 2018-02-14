#!/usr/bin/env bash

for f in `find . -name "*.png"`
do
    convert ${f} -resize 150x150 ${f}
done

for category in */ ; do
    echo "extracting for category $category"
    mkdir "../tu-test/${category}"

    i=0
    for x in ${category}/*; do
      if [ "$i" = 10 ]; then break; fi
      mv -- "${x}" "../tu-test/${category}"
      i=$((i+1))
    done


done