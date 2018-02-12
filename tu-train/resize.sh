#!/usr/bin/env bash

for f in `find . -name "*.png"`
do
    convert ${f} -resize 150x150 ${f}
done

for category in */ ; do
    echo "extracting for category $category"
    mkdir ../test/${category}

    i=0
    for x in ${category}/*; do
      if [ "$i" = 10 ]; then break; fi
      mv -- "${x}" ../test/${category}
      i=$((i+1))
    done


done