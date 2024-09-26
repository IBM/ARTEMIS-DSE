#!/bin/bash
set -e

original_dir="./Project_FARSI_orig"
modified_dir="./Project_FARSI"
diff_output_dir="./diffs"

mkdir -p "$diff_output_dir"

if [[ ! -d "$original_dir" ]]; then
    echo "Error: $original_dir does not exist!"
    exit 1
fi

find "$modified_dir" -type f | while read -r modified_file; do
    rel_path="${modified_file#$modified_dir/}"
    original_file="$original_dir/$rel_path"
    diff_file="$diff_output_dir/$rel_path.diff"
    mkdir -p "$(dirname "$diff_file")"
    diff -Naur "$original_file" "$modified_file" | sed -re '1,2 s/\t.*//' > "$diff_file"
    if [ ! -s "$diff_file" ]; then
        rm "$diff_file"
    fi
done
rm $diff_output_dir/.git.diff
echo "Successfully updated diff files in $diff_output_dir."
