#!/bin/sh

timestamp=`date +%Y-%m-%dT%H%M%S%z`
(
    cd ~/Documents/Neo4j/
    tar -cJf "/Users/corajr/Dropbox/masters/graph backups/$timestamp.tar.xz" soundcloud.graphdb
)
