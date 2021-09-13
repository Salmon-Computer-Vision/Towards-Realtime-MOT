# -*- mode: python ; coding: utf-8 -*-

import os.path as osp
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
conda_base = "C:/Users/samim/Anaconda3/envs/mot/Library/bin/"


a = Analysis(['count.py'],
             pathex=["C:\\Users\\samim\\Documents\\CMPT\\CMPT Master's Thesis\\Towards-Realtime-MOT"],
             binaries=[
                (osp.join(conda_base, 'cublas64_10.dll'), '.'),
                (osp.join(conda_base, 'cublasLt64_10.dll'), '.'),
                (osp.join(conda_base, 'cusolver64_10.dll'), '.'),
                (osp.join(conda_base, 'cusparse64_10.dll'), '.'),
                (osp.join(conda_base, 'nvToolsExt64_1.dll'), '.'),
                (osp.join(conda_base, 'nvrtc64_102_0.dll'), '.'),
                (osp.join(conda_base, 'cudart64_102.dll'), '.'),
                (osp.join(conda_base, 'cufft64_10.dll'), '.'),
                (osp.join(conda_base, 'cufftw64_10.dll'), '.'),
                (osp.join(conda_base, 'curand64_10.dll'), '.'),
                (osp.join(conda_base, 'uv.dll'), '.'),
                ],
             datas=[*collect_data_files("utils.utils", include_py_files=True),],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['PyQt5'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='count',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='count')
