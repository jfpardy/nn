#!/bin/sh
# ./build.sh        run the XOR example
# ./build.sh test   run the tests
set -e
cd "$(dirname "$0")"
mkdir -p build
case "$1" in
	test) odin test tests -out:build/nn_tests -vet -strict-style ;;
	*)    odin run examples/xor -out:build/xor -vet -strict-style ;;
esac
