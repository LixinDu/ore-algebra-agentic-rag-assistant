#!/usr/bin/env bash
set -e

MESSAGE="${1:-Update snapshot}"

git checkout --orphan public-snapshot
git add -A
git commit -m "$MESSAGE"
git push public public-snapshot:main --force
git checkout main
git branch -D public-snapshot

echo "Done. Public repo updated with a clean single commit: \"$MESSAGE\""
