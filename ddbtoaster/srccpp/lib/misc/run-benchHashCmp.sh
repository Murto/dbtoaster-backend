#!/bin/sh

rm -f ./mmap
g++ -O3 -std=c++17 benchHashCmp.cpp ../hash.cpp -I .. -I ../../driver -o benchHashCmp && ./benchHashCmp