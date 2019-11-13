#!/bin/sh

echo "starting grid search...\n"

echo "Learning rate = 0.00001"
python flappyBirdQ.py 0.00001

echo "Learning rate = 0.0001"
python flappyBirdQ.py 0.0001

echo "Learning rate = 0.001"
python flappyBirdQ.py 0.001

echo "Learning rate = 0.01"
python flappyBirdQ.py 0.01

echo "Learning rate = 0.1"
python flappyBirdQ.py 0.1

echo "grid search done..."
