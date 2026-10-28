"""Generate HTML API documentation, preferring pdoc and falling back to stdlib pydoc.

Generated HTML describes the modules in the package. When the three additional
model scripts are added, list their module names in MODULES below.
"""
from pathlib import Path
import importlib
import subprocess
import sys
import os
import pydoc

MODULES = ['data_processing', 'decision_tree', 'occupancy_utils', 'run_tests']
OPTIONAL_MODULES = ['linear_regression', 'historical_average', 'support_vector_regression']


def main() -> int:
    """Write docs/*.html for available modules and create a browsable index."""
    root = Path(__file__).resolve().parent
    docs = root / 'docs'
    docs.mkdir(exist_ok=True)
    modules = MODULES + [m for m in OPTIONAL_MODULES if (root / f'{m}.py').exists()]
    try:
        import pdoc  # noqa: F401
    except ImportError:
        # Offline fallback, uses Python's standard library.
        current_dir = Path.cwd()
        try:
            os.chdir(docs)
            sys.path.insert(0, str(root))
            for name in modules:
                pydoc.writedoc(importlib.import_module(name))
        finally:
            os.chdir(current_dir)
        engine = 'Python standard-library pydoc (offline fallback)'
    else:
        cmd = [sys.executable, '-m', 'pdoc', *modules, '-o', str(docs)]
        result = subprocess.call(cmd, cwd=root)
        if result:
            return result
        engine = 'pdoc'
    links = '\n'.join(f'<li><a href="{name}.html">{name}</a></li>' for name in modules)
    (docs / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>MMA3001 Code Documentation</title>'
        '<style>body{font:16px system-ui;max-width:850px;margin:60px auto;line-height:1.6;padding:0 18px}'
        'a{color:#155ca8}li{margin:9px 0}</style><h1>MMA3001 Code Documentation</h1>'
        '<p>Generated from source-code docstrings. Generator: ' + engine + '</p>'
        '<ul>' + links + '</ul>'
        '<p>For full model source documentation, add your three alternative-model .py files '
        'and rerun <code>python generate_docs.py</code>.</p></html>', encoding='utf-8')
    print(f'Generated {len(modules)} module pages and docs/index.html using {engine}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
