#!/usr/bin/env bash

if [ -e "pbn" ]; then
    echo "Directory 'pbn' already exists" >&2
    exit 1
fi

kaggle competitions download painter-by-numbers -f train.zip || exit 1

mkdir -p pbn
unzip train.zip -d pbn || exit 1

# These cause issues with TF for whatever reason.
for filename in \
    98873.jpg   81823.jpg   95347.jpg   92899.jpg   91033.jpg   95010.jpg   \
    79499.jpg   33557.jpg   50420.jpg   36600.jpg   72255.jpg   82594.jpg   \
    3917.jpg    13108.jpg   41945.jpg   101947.jpg  100486.jpg
do
    echo "   deleting: pbn/train/${filename}"
    rm -f "pbn/train/${filename}"
done

rm -f train.zip
