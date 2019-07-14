Installation
=============

hydrointerp can be installed via pip or conda::

  pip install hydrointerp

or::

  conda install -c mullenkamp hydrointerp

The core dependencies are `Pandas <http://pandas.pydata.org/pandas-docs/stable/>`_,  `Scipy <https://docs.scipy.org/doc/scipy/reference/index.html>`_, `xarray <http://xarray.pydata.org/en/stable/>`_, and `pyproj <http://pyproj4.github.io/pyproj/html/index.html>`_.

Be careful not to cause dependency conflicts due to pyproj. hydrointerp uses pyproj >= 2.1 due to a major improvement in functionality. Unfortunately, some other geographic python packages have not updated to the new version yet (e.g. geopandas). Consequently, you may not be able to have both hydrointerp and these other packages installed simultaneously until they update their package dependencies. 
