#!/usr/bin/env bash
set -euo pipefail

OWNER="${OWNER:-sliortega295-ops}"
OLD_REPO="${OLD_REPO:-CADense}"
NEW_REPO="${NEW_REPO:-CoFrame}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: GitHub CLI (gh) is required" >&2
  exit 1
fi

gh auth status

if gh repo view "$OWNER/$NEW_REPO" >/dev/null 2>&1; then
  echo "$OWNER/$NEW_REPO already exists; skipping rename."
else
  echo "Renaming $OWNER/$OLD_REPO -> $OWNER/$NEW_REPO"
  gh api --method PATCH "repos/$OWNER/$OLD_REPO" -f "name=$NEW_REPO" >/dev/null
fi

REMOTE_URL="https://github.com/$OWNER/$NEW_REPO.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

git push --set-upstream origin main
echo "Published: https://github.com/$OWNER/$NEW_REPO"
