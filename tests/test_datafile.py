import unittest
from gldas.connections import GldasRemote
from gldas.datafile import FileWithMeta

class TestDataFileRemote(unittest.TestCase):
    def setUp(self) -> None:
        url = "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/2006/001/GLDAS_NOAH025_3H.A20060101.0000.021.nc4"
        self.afile = FileWithMeta(url)

    def test_metadata(self):
        assert self.afile.metadata['VersionID'] == '2.1'
