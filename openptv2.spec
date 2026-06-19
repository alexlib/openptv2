# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for OpenPTV2 (TraitsUI desktop application).
Builds a standalone executable containing the GUI and both dual-engines (optv + python).
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules for the core packages and plugins
hiddenimports = (
    collect_submodules('openptv2') +
    collect_submodules('algorithms') +
    collect_submodules('gui.pyptv') +
    collect_submodules('gui.plugins') +
    [
        'PySide6',
        'traits',
        'traitsui',
        'chaco',
        'enable',
        'pyface',
        'pyface.ui.qt',
        'traitsui.qt',
        'traitsui.qt.extra',
        'numpy',
        'scipy',
        'yaml',
        'matplotlib',
        'pandas',
        'tables',
        'skimage',
        'tqdm',
        'imagecodecs',
        'flowtracks'
    ]
)

# Collect data files
datas = (
    collect_data_files('traitsui') +
    collect_data_files('pyface') +
    collect_data_files('enable') +
    collect_data_files('chaco')
)

# Add openptv2 package default configurations or directories if present
if os.path.exists('test_data'):
    datas.append(('test_data', 'test_data'))

a = Analysis(
    ['gui/pyptv/pyptv_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='openptv2-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Set to True for standard output/logging access
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='openptv2-gui',
)
