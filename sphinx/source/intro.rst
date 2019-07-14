Introduction
=============

There are many hydrological datasets that have a representation as two dimensional spatial data (e.. lon/lat or X/Y) over many time steps (e.g. gridded precipitation, wind, temperature, groundwater levels, etc.). These datasets may be derived from remote sensing equipment (e.g. NASA satellites) or may be derived from point observations (e.g. meteorological stations). For the purposes of hydrologic analysis, there are obvious needs to convert from point station data to gridded, gridded to points, gridded to different spatial resolution gridded, and so on. This package provides these conversions using different types of interpolators.

At the moment, these interpolators are purely 2D spatial interpolators and do not combine interpolations in time and space. Combining the two different dimension types (length and time) is not trivial nor is it obvious how that should be performed. As a consequence, each time step is interpolated independently.
