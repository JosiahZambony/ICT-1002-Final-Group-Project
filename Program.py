import os
import sys

# use os.system to compile c scripts and run c script
os.system('gcc algorithm.c -o execute');
string = './execute '
os.system(string);
