"""Locate body model files for both source checkouts and frozen app bundles."""
import os
import sys

MODELS_SUBDIR = os.path.join('data', 'body_models')


def _candidate_roots():
    """Folders that may hold data/body_models, most user-facing first."""
    roots = []
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        # macOS: <somewhere>/App.app/Contents/MacOS/exe -> <somewhere>
        if exe_dir.endswith(os.path.join('Contents', 'MacOS')):
            roots.append(os.path.dirname(os.path.dirname(os.path.dirname(exe_dir))))
        # Linux/Windows onedir: <somewhere>/App/exe -> <somewhere>/App
        roots.append(exe_dir)
        roots.append(getattr(sys, '_MEIPASS', exe_dir))  # files bundled at build time
    else:
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    env_root = os.environ.get('BMVIS_DATA_DIR')
    if env_root:
        roots.insert(0, env_root)
    return roots


def body_model_dir(name):
    """Return the folder for one model family (e.g. 'smpl'), preferring user-supplied files.

    The first existing candidate wins so a bundle can ship some models while
    users drop additional, individually licensed files next to the app.
    """
    for root in _candidate_roots():
        path = os.path.join(root, MODELS_SUBDIR, name)
        if os.path.isdir(path):
            return path
    return os.path.join(_candidate_roots()[-1], MODELS_SUBDIR, name)


def external_models_dir():
    """Where users should place extra model files for a frozen app."""
    return os.path.join(_candidate_roots()[0], MODELS_SUBDIR)
