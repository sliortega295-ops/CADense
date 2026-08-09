# Rename CADense to CoFrame and publish the initial implementation

The original `sliortega295-ops/CADense` repository was empty when this prototype was constructed. The local project and Python package are already named `CoFrame` / `coframe`.

From an authenticated machine with GitHub CLI:

```bash
cd CoFrame
gh auth login
bash scripts/publish_initial_repo.sh
```

The script performs the equivalent of:

```bash
gh api --method PATCH repos/sliortega295-ops/CADense -f name=CoFrame
git remote set-url origin https://github.com/sliortega295-ops/CoFrame.git
git push --set-upstream origin main
```

GitHub normally redirects the old repository URL after a rename, but experiment scripts and documentation should use the new canonical URL.
