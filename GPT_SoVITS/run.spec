# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

def collect_files(directory, dest_root):
    files = []
    for root, dirs, filenames in os.walk(directory):
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        
        for filename in filenames:
            if not filename.endswith('.py') and not filename.endswith('.pyc'):
                filepath = os.path.join(root, filename)
                # Calculate relative path from the source directory
                rel_path = os.path.relpath(filepath, directory)
                # Construct destination path
                dest_path = os.path.join(dest_root, os.path.dirname(rel_path))
                files.append((filepath, dest_path))
    return files

datas = []
datas += collect_data_files('budoux')
datas += collect_data_files('jieba_fast')

# Collect all non-py files from GPT_SoVITS and place them at root (for top-level imports like 'text', 'AR')
datas += collect_files('.', '.')

binaries = []
try:
    import _sqlite3
    sqlite_dll = os.path.join(os.path.dirname(_sqlite3.__file__), 'sqlite3.dll')
    if os.path.exists(sqlite_dll):
        binaries.append((sqlite_dll, '.'))
except ImportError:
    pass

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run',
)
