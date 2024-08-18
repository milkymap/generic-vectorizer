#!/bin/bash


# Exit immediately if a command exits with a non-zero status.
set -e

# Navigate to the src directory
cd generic_vectorizer

# Check if grpcio-tools is installed
if ! python -c "import grpc_tools" &> /dev/null; then
    echo "grpcio-tools is not installed. Installing now..."
    pip install grpcio-tools
fi

# Generate Python code from the proto file
python -m grpc_tools.protoc \
    -I./grpc_server/protos \
    --python_out=./grpc_server/interfaces \
    --grpc_python_out=./grpc_server/interfaces \
    --pyi_out=./grpc_server/interfaces \
    ./grpc_server/protos/strategies.proto

# Optional: Fix import statements in generated files
sed -i 's/import strategies_pb2/from . import strategies_pb2/' \
    ./grpc_server/interfaces/strategies_pb2_grpc.py

cd ../

echo "gRPC code generation completed successfully."