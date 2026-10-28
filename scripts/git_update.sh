#!/usr/bin/env bash
# Export aggregate results, validate the repository, commit safe paths, and push.
set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-$(git symbolic-ref --quiet --short HEAD)}"
MESSAGE="${1:-Update MemReason code and aggregate results ($(date +%F))}"
EXPECTED_REPOSITORY="${EXPECTED_REPOSITORY:-internbob/MemReason}"
REMOTE_URL="$(git config --get "remote.${REMOTE}.url")"

case "$REMOTE_URL" in
  *"${EXPECTED_REPOSITORY}"*) ;;
  *)
    echo "Refusing to push: ${REMOTE} points to ${REMOTE_URL}, expected ${EXPECTED_REPOSITORY}." >&2
    exit 1
    ;;
esac

echo "[1/7] Fetching ${REMOTE}/${BRANCH}"
git fetch "$REMOTE" "$BRANCH"
if ! git merge-base --is-ancestor "$REMOTE/$BRANCH" HEAD; then
  echo "Remote ${REMOTE}/${BRANCH} contains commits missing locally." >&2
  echo "Synchronize the branch before running this publishing script." >&2
  exit 1
fi

echo "[2/7] Exporting aggregate, de-identified results"
python scripts/export_public_results.py

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "[3/7] Running tests"
  python -m pytest -q
else
  echo "[3/7] Tests skipped because SKIP_TESTS=1"
fi

echo "[4/7] Staging the public allowlist"
git add -A -- \
  .gitignore \
  README.md \
  BUILD_STATUS.md \
  MemReason_RL_ADNI_Implementation_Spec.md \
  pyproject.toml \
  configs \
  memreason_rl \
  scripts \
  tests \
  reports/experiments \
  reports/methods \
  reports/results \
  reports/MemReason_monthly_weekly_report_202610.md \
  reports/MemReason_monthly_weekly_report_202610_en.md \
  reports/data_audit/summary.md \
  reports/data_audit/ida_scan_date_linkage.md \
  reports/data_audit/phase5c_memory_field_audit.md

echo "[5/7] Checking staged paths, file sizes, and common credential patterns"
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  case "$path" in
    outputs/*|processed/*|splits/*|*.pt|*.pth|*.ckpt|*.bin|*.npy|*.npz|*.h5|*.hdf5|*.parquet|*.pkl|*.pickle|*.nii|*.nii.gz|*.safetensors|idaSearch*.csv|adni_all_longitudinal_scales_visit_with_ravlt_logmem.csv|reports/matching/*.csv|reports/data_audit/*.csv)
      echo "Refusing forbidden staged path: $path" >&2
      exit 1
      ;;
  esac
  if [[ -f "$path" ]]; then
    bytes="$(wc -c < "$path")"
    if (( bytes > 20971520 )); then
      echo "Refusing staged file larger than 20 MiB: $path ($bytes bytes)" >&2
      exit 1
    fi
  fi
done < <(git diff --cached --name-only)

if git grep --cached -I -n -E \
  '(BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+|AKIA[0-9A-Z]{16})' -- .; then
  echo "Potential credential found in staged content; refusing to commit." >&2
  exit 1
fi

git diff --cached --check

echo "[6/7] Creating commit"
if git diff --cached --quiet; then
  echo "No public code or result changes to commit."
else
  git commit -m "$MESSAGE"
fi

echo "[7/7] Pushing ${BRANCH}"
git push -u "$REMOTE" "$BRANCH"
echo "Published to https://github.com/${EXPECTED_REPOSITORY}"
