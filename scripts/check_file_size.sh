#!/bin/sh
# Checks that each hand-written source file of the repository has at most
# 1,000 physical lines, as AGENTS.md requires. A file whose first five lines
# hold the marker of a generated file is exempt, as AGENTS.md states. Prints
# each file above the cap, and exits with a nonzero status when one exists.
# Run it from the root of the repository.

set -eu

# The script joins the marker from two halves, so that it does not hold the
# marker itself: tools such as the GitHub diff view hide a file that holds it.
git ls-files -z -- '*.rs' '*.c' '*.cpp' '*.h' '*.py' | xargs -0 awk '
    BEGIN { marker = "@" "generated" }
    FNR <= 5 && index($0, marker) { generated[FILENAME] = 1 }
    { lines[FILENAME] = FNR }
    END {
        status = 0
        for (file in lines) {
            if (!(file in generated) && lines[file] > 1000) {
                print file ": " lines[file] " lines, above the cap of 1,000"
                status = 1
            }
        }
        exit status
    }
'
