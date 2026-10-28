"""Run pytest and save a human-readable report and machine-readable JUnit XML."""
from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Execute test suite and write reports; return pytest exit status."""
    base = Path(__file__).resolve().parent
    out = base / 'results'
    out.mkdir(exist_ok=True)
    cmd = [sys.executable, '-m', 'pytest', '-v', '--junitxml', str(out / 'test_results.xml')]
    completed = subprocess.run(cmd, cwd=base, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'test_report.txt').write_text(completed.stdout, encoding='utf-8')
    print(completed.stdout)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
