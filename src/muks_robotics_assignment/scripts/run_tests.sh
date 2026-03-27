#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${PKG_DIR}/test_reports"
JUNIT_XML="${REPORT_DIR}/pytest_junit.xml"
OUTPUT_LOG="${REPORT_DIR}/pytest_output.txt"
SUMMARY_MD="${REPORT_DIR}/TEST_REPORT.md"

mkdir -p "${REPORT_DIR}"

echo "[INFO] Running pytest for ${PKG_DIR}"
python3 -m pytest "${PKG_DIR}/test" --junitxml="${JUNIT_XML}" | tee "${OUTPUT_LOG}"

echo "[INFO] Generating markdown report"
python3 "${PKG_DIR}/scripts/generate_test_report.py" "${JUNIT_XML}" "${SUMMARY_MD}"

echo "[INFO] Done"
echo "  - Log:    ${OUTPUT_LOG}"
echo "  - JUnit:  ${JUNIT_XML}"
echo "  - Report: ${SUMMARY_MD}"
