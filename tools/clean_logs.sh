#!/bin/bash
BASE="$1"

if [[ -z "$BASE" ]]; then
  echo "Usage: clean_logs.sh /projects/.../CA"
  exit 1
fi

find "$BASE/logs" -type f -mtime +333 -delete

