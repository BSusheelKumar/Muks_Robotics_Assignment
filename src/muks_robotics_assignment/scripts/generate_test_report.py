#!/usr/bin/env python3
"""Generate a compact markdown test report from pytest JUnit XML."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def _as_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def build_report(junit_xml: Path, output_md: Path) -> None:
    root = ET.parse(junit_xml).getroot()

    if root.tag == 'testsuites':
        tests = _as_int(root.attrib.get('tests'))
        failures = _as_int(root.attrib.get('failures'))
        errors = _as_int(root.attrib.get('errors'))
        skipped = _as_int(root.attrib.get('skipped'))
        time_sec = float(root.attrib.get('time') or 0.0)

        # Some pytest XML outputs keep totals only at <testsuite> level.
        if tests == 0 and len(root):
            tests = sum(_as_int(ts.attrib.get('tests')) for ts in root.findall('testsuite'))
            failures = sum(_as_int(ts.attrib.get('failures')) for ts in root.findall('testsuite'))
            errors = sum(_as_int(ts.attrib.get('errors')) for ts in root.findall('testsuite'))
            skipped = sum(_as_int(ts.attrib.get('skipped')) for ts in root.findall('testsuite'))
            time_sec = sum(float(ts.attrib.get('time') or 0.0) for ts in root.findall('testsuite'))
    else:
        tests = _as_int(root.attrib.get('tests'))
        failures = _as_int(root.attrib.get('failures'))
        errors = _as_int(root.attrib.get('errors'))
        skipped = _as_int(root.attrib.get('skipped'))
        time_sec = float(root.attrib.get('time') or 0.0)

    passed = max(tests - failures - errors - skipped, 0)
    status = 'PASS' if failures == 0 and errors == 0 else 'FAIL'

    lines = [
        '# Automated Test Report',
        '',
        f'- Generated at: `{datetime.utcnow().isoformat()}Z`',
        f'- Overall status: **{status}**',
        f'- Total tests: **{tests}**',
        f'- Passed: **{passed}**',
        f'- Failed: **{failures}**',
        f'- Errors: **{errors}**',
        f'- Skipped: **{skipped}**',
        f'- Duration: **{time_sec:.3f} s**',
        '',
        '## Notes',
        '',
        '- This report is generated from `pytest --junitxml` output.',
        '- For full details, check `test_reports/pytest_output.txt`.',
    ]

    output_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: generate_test_report.py <junit_xml> <output_md>')
        return 2

    junit_xml = Path(sys.argv[1]).resolve()
    output_md = Path(sys.argv[2]).resolve()
    if not junit_xml.exists():
        print(f'JUnit file not found: {junit_xml}')
        return 1

    output_md.parent.mkdir(parents=True, exist_ok=True)
    build_report(junit_xml, output_md)
    print(f'Wrote report: {output_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
