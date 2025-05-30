#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: ./download_from_s3.sh <filename> [output_path]"
    echo "Example: ./download_from_s3.sh raw_data/20250527_111205_test_data.csv"
    exit 1
fi

INPUT_PATH=$1
WORK_FOLDER=${HPC_WORK_FOLDER:-""}
if [ -z "$WORK_FOLDER" ]; then
    echo "Error: HPC_WORK_FOLDER environment variable is not set"
    exit 1
fi

OUTPUT_PATH="${WORK_FOLDER}/$(basename "$INPUT_PATH")"

BUCKET_URL=${S3_BUCKET_URL:-""}
if [ -z "$BUCKET_URL" ]; then
    echo "Error: S3_BUCKET_URL environment variable is not set"
    exit 1
fi

FULL_URL="${BUCKET_URL}/${INPUT_PATH}"

echo "Downloading ${INPUT_PATH} from S3..."
echo "Full URL: ${FULL_URL}"
echo "Saving to: ${OUTPUT_PATH}"
if curl -f -s -o "$OUTPUT_PATH" "$FULL_URL"; then
    echo "Successfully downloaded to $OUTPUT_PATH"
    exit 0
else
    echo "Failed to download file"
    exit 1
fi 