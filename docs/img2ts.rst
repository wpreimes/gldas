Conversion to time series format
================================

For a lot of applications it is favorable to convert the image based format into
a format which is optimized for fast time series retrieval. This is what we
often need for e.g. validation studies. This can be done by stacking the images
into a netCDF file and choosing the correct chunk sizes or a lot of other
methods. We have chosen to do it in the following way:

- Store only the reduced gaußian grid points since that saves space.
- Further reduction the amount of stored data by saving only land points if selected.
- Store the time series in netCDF4 in the Climate and Forecast convention
  `Orthogonal multidimensional array representation
  <http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#_orthogonal_multidimensional_array_representation>`_
- Store the time series in 5x5 degree cells. This means there will be 2566 cell
  files (without reduction to land points) and a file called ``grid.nc``
  which contains the information about which grid point is stored in which file.
  This allows us to read a whole 5x5 degree area into memory and iterate over the time series quickly.

  .. image:: 5x5_cell_partitioning.png
     :target: _images/5x5_cell_partitioning.png

This conversion can be performed using the ``gldas_repurpose`` command line
program. An example would be:

.. code-block:: shell

   gldas_repurpose /download/image/path /output/timeseries/path 2000-01-01 2001-01-01 SoilMoi0_10cm_inst SoilMoi10_40cm_inst

Which would take GLDAS Noah data stored in ``/gldas_data`` from January 1st
2000 to January 1st 2001 and store the parameters for the top 2 layers of soil moisture as time
series in the folder ``/timeseries/data``.

Conversion to time series is performed by the `repurpose package
<https://github.com/TUW-GEO/repurpose>`_ in the background. For custom settings
or other options see the `repurpose documentation
<http://repurpose.readthedocs.io/en/latest/>`_ and the code in
``gldas.reshuffle``.

**Note**: If a ``RuntimeError: NetCDF: Bad chunk sizes.`` appears during reshuffling, consider downgrading the
netcdf4 library via:

.. code-block:: shell

  conda install -c conda-forge netcdf4=1.2.2

Reading converted time series data
----------------------------------

For reading the data the ``gldas_repurpose`` command produces the class
``GLDASTs`` can be used:

.. code-block:: python

    from gldas.interface import GLDASTs
    ds = GLDASTs(ts_path, ioclass_kws={'read_bulk': True})
    # read_ts takes either lon, lat coordinates or a grid point indices.
    # and returns a pandas.DataFrame
    ts = ds.read(45, 15)

    >>> ts
                         Snowf_tavg  ...  SoilTMP100_200cm_inst
    2000-01-01 03:00:00         0.0  ...             292.014526
    2000-01-01 06:00:00         0.0  ...             292.006256
    2000-01-01 09:00:00         0.0  ...             291.998505
    2000-01-01 12:00:00         0.0  ...             291.981201
    2000-01-01 15:00:00         0.0  ...             291.974579
    ...                         ...  ...                    ...
    2023-10-31 09:00:00         0.0  ...             299.025757
    2023-10-31 12:00:00         0.0  ...             299.025024
    2023-10-31 15:00:00         0.0  ...             299.014282
    2023-10-31 18:00:00         0.0  ...             299.003540
    2023-10-31 21:00:00         0.0  ...             298.992798