

# GLDAS netcdf filename conventions, tried in order
fn_templates = [
    "{shortname}.A{date}.{time}.{vers_id}.{ext}",
    "{shortname}.A{date}.{vers_id}.{ext}",
    ]

gldas_url = "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/"
AUTH_HOST = 'urs.earthdata.nasa.gov'

class DataDirError(IOError):
    def __int__(self, directory):
        self.directory = directory
        self.message = f"Directory {self.directory} not found"
        super(DataDirError, self).__int__(self.message)

