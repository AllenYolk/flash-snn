import functools
from pathlib import Path
import atexit
import threading

_CLEANUP_TMP_PYTHON_FILES_REGISTERED = False
_CLEANUP_TMP_PYTHON_FILES_REGISTERED_LOCK = threading.Lock()


def cleanup_tmp_python_files():
    print("Cleaning up temporary python files!")
    for f in Path("/tmp").glob("*.py"):
        try:
            f.unlink(missing_ok=True)
        except Exception as e:
            pass  # ignore errors


def ensure_cleanup_tmp_python_files(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _CLEANUP_TMP_PYTHON_FILES_REGISTERED
        with _CLEANUP_TMP_PYTHON_FILES_REGISTERED_LOCK:
            if not _CLEANUP_TMP_PYTHON_FILES_REGISTERED:
                atexit.register(cleanup_tmp_python_files)
                _CLEANUP_TMP_PYTHON_FILES_REGISTERED = True
        return fn(*args, **kwargs)

    return wrapper
