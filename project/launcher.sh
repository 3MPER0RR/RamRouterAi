#!/bin/bash

clear

echo "======================="
echo " CORTEXFLOW LAUNCHER "
echo "======================="
echo
echo "1) Standard Runtime"
echo "2) RAM Runtime"
echo "3) Offline Mode"
echo "4) Exit"
echo

read -p "> " choice

case $choice in

1)
    python3 main.py
    ;;

2)
    python3 experimental/ram_runtime.py
    ;;

3)
    OFFLINE=1 python3 main.py
    ;;

4)
    exit
    ;;

*)
    echo "Invalid option"
    ;;
esac
