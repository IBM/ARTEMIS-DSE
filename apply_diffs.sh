#!/bin/bash
set -e

original_dir="./Project_FARSI"
diff_dir="./diffs"

find "$diff_dir" -type f -name "*.diff" | while read -r diff_file; do
    rel_path="${diff_file#$diff_dir/}"
    rel_path="${rel_path%.diff}"

    original_file="$original_dir/$rel_path"
    original_file_dir=$(dirname "$original_file")

    if grep -q "Adding new file:" "$diff_file"; then
        mkdir -p "$original_file_dir"
        cp "${diff_file%.diff}" "$original_file"
    else
        mkdir -p "$original_file_dir"

        patch "$original_file" < "$diff_file"
    fi
done
echo "Successfully patched all files in $original_dir."