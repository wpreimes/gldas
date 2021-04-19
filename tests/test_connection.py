from gldas.connections import GldasLocal, GldasRemote
import unittest
from tests import testdata_path
from datetime import datetime

class TestRemoteConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.remote = GldasRemote()  # no dataset assigned
        cls.testdataset = 'GLDAS_NOAH025_3H.2.1'

    def setUp(self) -> None:
        self.remote.connect(self.testdataset)
        assert self.testdataset in self.remote._datasets

    def test_available_datasets(self):
        assert 'GLDAS_NOAH025_3H_EP.2.1' in self.remote._datasets
        assert 'GLDAS_NOAH025_3H.2.0' in self.remote._datasets
        assert self.remote.dataset == self.testdataset

    def test_get_first_last_item(self):
        first, last = self.remote.get_first_last_item(('2015', '001'), '*.nc4')
        assert first == 'GLDAS_NOAH025_3H.A20150101.0000.021.nc4'
        assert last == 'GLDAS_NOAH025_3H.A20150101.2100.021.nc4'

    def test_list_folders(self):
        folders = self.remote.list_folders(exclude=('doc',))
        assert folders[0] == '2000'

    def test_list_folders_years(self):
        assert self.remote.list_folders_years()[0] == 2000

    def test_list_folders_days_for_year(self):
        assert self.remote.list_folders_days_for_year(2015) == list(range(1, 366))

    def test_list_files_for_day(self):
        assert self.remote.list_files_for_day(2015, 1, extension='.nc4')\
               == [f'GLDAS_NOAH025_3H.A20150101.{h:02}00.021.nc4' for h in range(0, 24, 3)]

    def test_parse_filename(self):
        f_cont = self.remote.parse_filename('first')
        l_cont = self.remote.parse_filename('last')

        assert isinstance(f_cont['datetime'], datetime)
        assert isinstance(l_cont['datetime'], datetime)
        assert f_cont['shortname'] == l_cont['shortname']
        assert f_cont['version'] == l_cont['version'] == '021'
        assert f_cont['ext'] == l_cont['ext'] == 'nc4'

class TestLocalConnection(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        rootpath = testdata_path / 'GLDAS_NOAH_image_data'
        cls.local = GldasLocal(path=rootpath)

    def test_get_first_last_item(self):
        first, last = self.local.get_first_last_item(('2015', '001'))
        assert first == 'GLDAS_NOAH025_3H.A20150101.0000.021.nc4'
        assert last == 'GLDAS_NOAH025_3H.A20150101.0000.021.nc4.xml'

    def test_list_folders(self):
        folders = self.local.list_folders(exclude=('doc',))
        assert folders == ['2015']

    def test_list_folders_years(self):
        assert self.local.list_folders_years() == [2015]

    def test_list_folders_days_for_year(self):
        assert self.local.list_folders_days_for_year(2015, as_int=False) == ['001']

    def test_list_files_for_day(self):
        assert self.local.list_files_for_day(2015, 1, extension='.nc4')\
               == ['GLDAS_NOAH025_3H.A20150101.0000.021.nc4']

    def test_parse_filename(self):
        f_cont = self.local.parse_filename('first')
        l_cont = self.local.parse_filename('last')

        assert isinstance(f_cont['datetime'], datetime)
        assert isinstance(l_cont['datetime'], datetime)
        assert f_cont['shortname'] == l_cont['shortname']
        assert f_cont['version'] == l_cont['version'] == '021'
        assert f_cont['ext'] == l_cont['ext'] == 'nc4'