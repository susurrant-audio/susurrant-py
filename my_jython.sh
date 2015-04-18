#!/bin/sh

CP="/Users/chrisjr/Development/susurrant_prep/susurrant-utils/lib/sis-jhdf5-batteries_included.jar"
CP="${CP}:/Users/chrisjr/Downloads/mallet-2.0.7/dist/mallet.jar"
CP="${CP}:/Users/chrisjr/Downloads/mallet-2.0.7/dist/mallet-deps.jar"

jython -J-cp "$CP" -J-Xmx4g "$@"
