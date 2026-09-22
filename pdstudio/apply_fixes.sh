#!/usr/bin/env bash
# PD Insight Studio: repo cleanup.
# Run this from INSIDE your local clone of PD-Insight-Studio.
# It renames the real app to main.py, retires the two dead PySide6 files,
# and drops in the corrected README, requirements, licence, gitignore and sample data.
#
# Nothing is deleted permanently: the dead files move to experiments/ and
# everything is staged for you to review before you commit.

set -euo pipefail

if [ ! -f "main-DESKTOP-MQ8GDGD.py" ]; then
  echo "ERROR: run this from inside your PD-Insight-Studio clone."
  echo "Expected to find main-DESKTOP-MQ8GDGD.py here."
  exit 1
fi

FIXES="${1:-}"
if [ -z "$FIXES" ] || [ ! -f "$FIXES/README.md" ]; then
  echo "Usage: bash apply_fixes.sh /path/to/repo-fixes"
  echo "  (the folder containing README.md, requirements.txt, LICENSE, .gitignore, sample_synthetic_tapping.csv)"
  exit 1
fi

echo "==> Retiring the two PySide6 dead ends to experiments/"
mkdir -p experiments
git mv main.py       experiments/prototype_pyside_v1.py 2>/dev/null || mv -n main.py       experiments/prototype_pyside_v1.py
git mv main2.py      experiments/prototype_pyside_v2.py 2>/dev/null || mv -n main2.py      experiments/prototype_pyside_v2.py

echo "==> Promoting the real application to main.py"
git mv main-DESKTOP-MQ8GDGD.py main.py 2>/dev/null || mv -n main-DESKTOP-MQ8GDGD.py main.py

echo "==> Removing the stray Windows backup requirements file"
git rm -q --cached requirements-DESKTOP-MQ8GDGD.txt 2>/dev/null || true
mkdir -p experiments && mv -n requirements-DESKTOP-MQ8GDGD.txt experiments/ 2>/dev/null || true

echo "==> Removing the empty sample CSV (headers only, no rows)"
git rm -q mpower_spss_ready_kept.csv 2>/dev/null || rm -f mpower_spss_ready_kept.csv

echo "==> Installing corrected files"
cp "$FIXES/README.md"                       ./README.md
cp "$FIXES/requirements.txt"                ./requirements.txt
cp "$FIXES/LICENSE"                         ./LICENSE
cp "$FIXES/.gitignore"                      ./.gitignore
cp "$FIXES/sample_synthetic_tapping.csv"    ./sample_synthetic_tapping.csv

cat > experiments/README.md <<'EOF'
# Experiments

Earlier PySide6 prototypes, kept for history. They do not import the analysis
modules and are not the application. The application is `main.py` in the repo root.
EOF

echo "==> Staging"
git add -A

echo
echo "Done. Review with:  git status  and  git diff --cached"
echo "Then:"
echo "  git commit -m 'Clean up repo: promote real app to main.py, fix requirements, add licence, synthetic sample data and honest README'"
echo "  git push"
echo
echo "After pushing, open the five issues in HELP_WANTED_ISSUES.md and add the topic"
echo "tags on GitHub: parkinsons, healthcare, data-analysis, python, real-world-data."
