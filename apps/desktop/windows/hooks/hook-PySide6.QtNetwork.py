"""PyInstaller QtNetwork hook without the optional run-time SSL probe.

PyInstaller's stock hook calls ``QSslSocket.supportsSsl()`` in an isolated
worker to discover external OpenSSL DLLs. That Qt call can block indefinitely
on some Windows/PySide combinations. MicroSeg Desktop does not use Qt network
or TLS APIs, but Qt's linked module graph still causes this hook to run. Keep
the normal Qt module/plugin dependencies and omit only the unused SSL probe.
"""

from PyInstaller.utils.hooks.qt import add_qt6_dependencies


hiddenimports, binaries, datas = add_qt6_dependencies(__file__)
