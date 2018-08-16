#!/bin/bash

# Enable recursive glob patterns
shopt -s globstar

# Path to main data folder
DATA_PATH=./data
echo "Path to documents is $DATA_PATH"

# Delete all files that we can't use because of task formulation
# echo "Deleting all unnecessary files..."
# find ${DATA_PATH} -not -name 'en.raw' -not -name 'en.tags' -type f -delete

# Delete previous experiments
#echo "Delete previous target..."
#find ${DATA_PATH} -name 'target.tsv' -type f -delete

# Extract target from each document's folder
parse_ner () {
    local i=${1}
    # Delete all empty lines (sentences delimiters)
    sed -i '/^\s*$/d' ${i}
    # Parse out token, pos and target (named entity type)
    local parent_dir="$(dirname ${i})"
    local doc="$(basename "$parent_dir")"
    local p="$(basename "$(dirname "$parent_dir")")"
    local target="$parent_dir"/target.tsv
    join -1 4 -2 1 "$parent_dir"/en.tok.off "${i}" | \
    awk -v par="$p" -v d="$doc" '{OFS="\t"};{print $1, $4, $5, $7, par, d}' >> ${target}
}

export -f parse_ner

find ${DATA_PATH} -name 'en.tags' -type f | parallel --eta parse_ner

# Aggregate all raw documents to one file
OUTPUT=${DATA_PATH}/agg.tsv
OUT_HEADER="token\tposition\tpos\tner\tpart\tdocument"
echo -e "Created aggregation file $OUTPUT with header $OUT_HEADER"
echo -e "$OUT_HEADER" > "$OUTPUT"
#
append_to_out() {
    local out="$1"
    local j="$2"
    awk '{printf "%s\n", $0}' ${j} >> "$out"
}
export -f append_to_out
#
find ${DATA_PATH} -name 'target.tsv' -type f | parallel --eta append_to_out ${OUTPUT}