

# GLDAS netcdf filename convention
data_fn_templ = "{shortname}.A{date}.{time}.{version}.{ext}"
data_fn_templ_dt_format = '%Y%m%d.%H%M'


class DataDirError(IOError):
    def __int__(self, directory):
        self.directory = directory
        self.message = f"Directory {self.directory} not found"
        super(DataDirError, self).__int__(self.message)

