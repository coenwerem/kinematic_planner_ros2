import sys as _sys

# An apt-installed python3-matplotlib (3.6.3, an old-style setuptools
# namespace package with a *-nspkg.pth site shim) can bind sys.modules
# ["mpl_toolkits"] to /usr/lib/python3/dist-packages before this package
# ever runs, even when a newer matplotlib is pip-installed. That stale
# binding lacks mplot3d.plotvol3, which spatialmath.base.graphics imports
# unconditionally, so any "import spatialmath" downstream fails with
# ImportError: cannot import name 'plotvol3'. Dropping the pre-bound
# entries here forces Python to re-resolve mpl_toolkits against the
# correct (usually pip-installed) matplotlib on first import.
_stale_names = [n for n in _sys.modules if n == "mpl_toolkits" or n.startswith("mpl_toolkits.")]
for _name in _stale_names:
    del _sys.modules[_name]
del _sys, _stale_names
