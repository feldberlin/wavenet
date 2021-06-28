import contextlib
import shutil
import tempfile
from pathlib import Path

# base director for fixtures
fixtures = Path("fixtures")


@contextlib.contextmanager
def tempdir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)
