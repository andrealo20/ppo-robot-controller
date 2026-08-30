"""Empty on purpose.

Its only job is to sit at the repository root: pytest adds the directory
holding conftest.py to sys.path when it imports it, which is what lets test
modules do `from src.x import y` regardless of the directory pytest was
invoked from. Without this file, `src` is only importable when the process
happens to be started with the repository root already on sys.path (e.g. via
`python -m src.train`), which pytest's own collection does not guarantee.
"""
