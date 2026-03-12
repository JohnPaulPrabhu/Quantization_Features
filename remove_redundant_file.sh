#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./dedup_models.sh /path/to/shared_workspace
#
# Example:
#   ./dedup_models.sh /data/model_workspaces

ROOT_DIR="${1:-}"

if [[ -z "$ROOT_DIR" ]]; then
    echo "Usage: $0 /path/to/root_folder"
    exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "Error: '$ROOT_DIR' is not a valid directory"
    exit 1
fi

LOG_FILE="$ROOT_DIR/dedup_$(date +%Y%m%d_%H%M%S).log"
TMP_FILE="$(mktemp)"
DUP_FILE="$(mktemp)"

echo "Starting deduplication at $(date)" | tee -a "$LOG_FILE"
echo "Root directory: $ROOT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Step 1: Find files and generate md5 + filepath
# -type f    => real files only
# ! -xtype l => ignore broken symlinks if any
find "$ROOT_DIR" -type f -print0 | while IFS= read -r -d '' file; do
    # Skip log files created by this script
    if [[ "$file" == *.log ]]; then
        continue
    fi

    checksum=$(md5sum "$file" | awk '{print $1}')
    printf '%s\t%s\n' "$checksum" "$file"
done > "$TMP_FILE"

# Step 2: Group by checksum and process duplicates
awk -F '\t' '
{
    count[$1]++
    files[$1] = files[$1] ORS $2
}
END {
    for (c in count) {
        if (count[c] > 1) {
            print "CHECKSUM=" c
            print files[c]
            print "---"
        }
    }
}
' "$TMP_FILE" > "$DUP_FILE"

if [[ ! -s "$DUP_FILE" ]]; then
    echo "No duplicate files found." | tee -a "$LOG_FILE"
    rm -f "$TMP_FILE" "$DUP_FILE"
    exit 0
fi

current_checksum=""
declare -a current_files=()

process_group() {
    local checksum="$1"
    shift
    local files=("$@")

    if [[ ${#files[@]} -le 1 ]]; then
        return
    fi

    # Keep the first file as the original
    local original="${files[0]}"

    echo "Duplicate group found:" | tee -a "$LOG_FILE"
    echo "  Checksum: $checksum" | tee -a "$LOG_FILE"
    echo "  Keeping : $original" | tee -a "$LOG_FILE"

    for ((i=1; i<${#files[@]}; i++)); do
        local duplicate="${files[$i]}"

        # Extra safety: skip if same path
        if [[ "$duplicate" == "$original" ]]; then
            continue
        fi

        # Remove duplicate and replace with symlink
        rm -f "$duplicate"
        ln -s "$original" "$duplicate"

        echo "  Replaced: $duplicate -> $original" | tee -a "$LOG_FILE"
    done

    echo "" | tee -a "$LOG_FILE"
}

while IFS= read -r line; do
    if [[ "$line" == CHECKSUM=* ]]; then
        # Process previous group if exists
        if [[ -n "$current_checksum" && ${#current_files[@]} -gt 0 ]]; then
            process_group "$current_checksum" "${current_files[@]}"
        fi

        current_checksum="${line#CHECKSUM=}"
        current_files=()
    elif [[ "$line" == "---" ]]; then
        if [[ -n "$current_checksum" && ${#current_files[@]} -gt 0 ]]; then
            process_group "$current_checksum" "${current_files[@]}"
        fi
        current_checksum=""
        current_files=()
    elif [[ -n "$line" ]]; then
        current_files+=("$line")
    fi
done < "$DUP_FILE"

# Final cleanup
rm -f "$TMP_FILE" "$DUP_FILE"

echo "Deduplication completed at $(date)" | tee -a "$LOG_FILE"