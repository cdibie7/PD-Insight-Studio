# Do this, step by step

**Do not delete the repo. Do not delete any .py files.** Your code is the product. This only changes file *names* and *adds* packaging files.

---

## Before you start

Open a terminal in your local PD-Insight-Studio folder. On Windows, right-click the folder and choose "Open Git Bash here", or use the terminal in VS Code.

Make sure you are up to date and nothing is uncommitted:

```bash
git status
git pull
```

Then make a safety branch, so the current state is always recoverable:

```bash
git checkout -b repo-cleanup
```

If anything goes wrong at any point, `git checkout main` puts you back exactly where you were.

---

## Step 1: put the five new files into the folder

Download these from the chat and drop them into the PD-Insight-Studio folder, replacing the existing README and requirements:

- `README.md`  (replaces yours)
- `requirements.txt`  (replaces yours)
- `LICENSE`  (new)
- `.gitignore`  (new)
- `sample_synthetic_tapping.csv`  (new)

`HELP_WANTED_ISSUES.md` and `apply_fixes.sh` are **not** for the repo. Keep them outside it.

---

## Step 2: fix the file names

Run these one at a time and read what each does.

```bash
# Move the two PySide6 dead ends out of the way. They are kept, not deleted.
mkdir experiments
git mv main.py  experiments/prototype_pyside_v1.py
git mv main2.py experiments/prototype_pyside_v2.py

# Promote your real application to main.py
git mv main-DESKTOP-MQ8GDGD.py main.py

# Retire the stray Windows backup requirements file, also kept
git mv requirements-DESKTOP-MQ8GDGD.txt experiments/

# Remove the empty sample CSV. It has a header row and zero data,
# so nothing is lost. The new synthetic file replaces it.
git rm mpower_spss_ready_kept.csv
```

---

## Step 3: check it still runs before committing

```bash
python -m venv .venv
source .venv/Scripts/activate     # Git Bash on Windows
pip install -r requirements.txt
python main.py
```

The app should open. Load `sample_synthetic_tapping.csv` on the Load Data tab and click through to QC. You should see rows flagged with reasons. If that works, the repo is correct.

**Do not commit until this passes.** If it fails, tell me the error rather than pushing.

---

## Step 4: commit and push

```bash
git add -A
git status          # read this before committing
git commit -m "Promote real app to main.py, fix requirements, add MIT licence, synthetic sample data and rewritten README"
git push -u origin repo-cleanup
```

Then on GitHub, open a pull request from `repo-cleanup` into `main` and merge it. Or if you would rather not bother with a PR:

```bash
git checkout main
git merge repo-cleanup
git push
```

---

## Step 5: the bit that actually attracts contributors

On GitHub, in the repo:

1. **Add topics.** Click the gear next to "About" and add: `parkinsons`, `healthcare`, `data-analysis`, `python`, `real-world-data`, `digital-health`. This is how people find it.
2. **Write the About line.** Something like: *Desktop app for exploratory analysis of Parkinson's disease real-world data, with transparent quality control.*
3. **Open the five issues** from `HELP_WANTED_ISSUES.md`. Create the labels `help wanted`, `good first issue`, `modelling`, `testing`, `interpretability` as you go. GitHub surfaces `help wanted` and `good first issue` in its own discovery feeds, which is the single cheapest way to get a stranger to look.
4. **Add a screenshot.** Run the app, screenshot the Insights tab with the sample data loaded, save it as `docs/screenshot.png`, and add `![](docs/screenshot.png)` near the top of the README. A repo with a picture gets meaningfully more attention than one without.

---

## What not to do

- Do not force-push over `main` without checking `git status` first.
- Do not delete the `.py` files. All seven are load-bearing.
- Do not commit any real participant data. The synthetic file exists so you never have to.
