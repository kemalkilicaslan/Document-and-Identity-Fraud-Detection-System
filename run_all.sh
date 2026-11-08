#!/usr/bin/env bash
# Usage: ./run_all.sh           run the tests and every pipeline step
#        ./run_all.sh 07        resume from step 07
# Python is taken from $PYTHON (default: python3).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[[ "$PY" == */* ]] && PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
FROM="${1:-00}"
FIG_PREFIX="Document-and-Identity-Fraud-Detection-"
mkdir -p reports/logs
LOG="reports/logs/run_$(date +%Y%m%d_%H%M%S).log"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

echo ">> unit tests" | tee -a "$LOG"
"$PY" -m pytest -q 2>&1 | tee -a "$LOG"

for s in scripts/[0-9][0-9]_*.py; do
  n="$(basename "$s" | cut -c1-2)"
  [[ "$n" < "$FROM" ]] && continue
  echo ">> $s" | tee -a "$LOG"
  (cd scripts && "$PY" "$(basename "$s")") 2>&1 | tee -a "$LOG"
done

for f in reports/figures/*.png reports/figures/*.jpg; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  name="$(echo "${base%.*}" | awk -F_ '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)} 1' OFS=-)"
  cp "$f" "${FIG_PREFIX}${name}.${base##*.}"
done

echo
echo "Done. Log: $LOG"
echo "Reports  : reports/management_report.html, reports/case_study.html"
echo "Dashboard: streamlit run src/docfraud/dashboard.py"
echo "API      : uvicorn docfraud.api:app --reload  ->  http://127.0.0.1:8000/docs"
