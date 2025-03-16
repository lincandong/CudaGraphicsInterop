set -e

if [ -z "$1" ]
then
	echo >&2 "No Path Tracer given.";
	exit 1;
fi

pt=$1

echo "Building project $pt"

cmake --build ./build --config Release --target $pt

./build/$pt/$pt $2 $3
