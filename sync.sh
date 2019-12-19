#!/bin/bash
while inotifywait -r -e modify,create,delete,move carsmoney; do
  rsync -r -a -v -e "ssh -oStrictHostKeyChecking=no -i~/.ssh/cs" --delete carsmoney dylan@35.202.94.149:code/CS682
done
