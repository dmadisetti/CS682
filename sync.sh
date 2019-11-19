#!/bin/bash
while inotifywait -r -e modify,create,delete,move carsmoney; do
  rsync -r -a -v -e "ssh -oStrictHostKeyChecking=no -i~/.ssh/cs -p$@" --delete carsmoney root@0.tcp.ngrok.io:/usr/local/lib/python3.6/dist-packages/
done
