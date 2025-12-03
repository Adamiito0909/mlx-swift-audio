#!/bin/bash
# Analyze code authorship using git blame

cd "$(git rev-parse --show-toplevel)"

git ls-files -- '*.swift' '*.m' '*.h' '*.c' '*.cpp' | while read -r file; do
    git blame --line-porcelain "$file" 2>/dev/null
done | grep "^author " | sed 's/^author //' | sort | uniq -c | sort -rn | awk '
    { lines[NR]=$1; total+=lines[NR]; $1=""; author[NR]=substr($0,2) }
    END {
        print "Lines of code by contributor:\n"
        for (i=1; i<=NR; i++) {
            printf "%6d %7s %s\n", lines[i], sprintf("(%.1f%%)", lines[i]/total*100), author[i]
        }
        printf "\nTotal: %d lines\n", total
    }'
