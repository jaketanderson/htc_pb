#!/bin/bash

set -e
ENVNAME=pb
ENVDIR=$ENVNAME
export PATH
mkdir $ENVDIR
tar -xzf /$ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

pwd

# Run python
mkdir -p working_data
echo $DELPHIEXEC
python3 worker.py "$1"

echo "Completed."
exit
