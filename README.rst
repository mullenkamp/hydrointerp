hydrointerp - A Python package for interpolating hydrologic data
===================================================================

The hydrointerp package includes several interpolation functions specifically designed for hydrologic data. These are mainly derrived from `Scipy <https://docs.scipy.org/doc/scipy/reference/index.html>`_ interpolation functions like `griddata <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html>`_.

.. Documentation
.. --------------
.. The primary documentation for the package can be found `here <http://hydrointerp.readthedocs.io>`_.

Installation
------------
hydrointerp can be installed via pip or conda::

  pip install hydrointerp

or::

  conda install -c mullenkamp hydrointerp

The core dependency are `Pandas <http://pandas.pydata.org/pandas-docs/stable/>`_,  `Scipy <https://docs.scipy.org/doc/scipy/reference/index.html>`_, `xarray <http://xarray.pydata.org/en/stable/>`_, and `pyproj <http://pyproj4.github.io/pyproj/html/index.html>`_.

Project Plan
------------
- Core functions for 1, 2, and possibly 3D interpolation methods particularly geared towards hyrologic data (e.g. precipitation, flow, groundwater levels, etc).
- Specific functions built upon the core interpolations functions for those hydrologic features mentioned above.
- Functions/classes for importing scientific data from external sources that could be used as part of the interpolations (e.g. NIWA, MetService, NASA).
- Associated utility functions
