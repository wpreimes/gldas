import urllib.error

import pandas as pd
import fnmatch
import abc
import os
from typing import Union
import re
from urllib.parse import urlparse, urljoin
from functools import wraps, reduce
from parse import parse
from datetime import datetime
import gldas.const as glob

def needs_dataset(func):
    @wraps(func)
    def _impl(self, *args, **kwargs):
        #if self.dataset is None:
        #    raise IOError("Not conncected to any dataset")
        print('test')
        return func(self, *args, **kwargs)
    return _impl


class DatasetConnection:
    """ Explores GLDAS data (remote or local) """

    def __init__(self,
                 root: str):

        self.root = root

    @abc.abstractproperty
    def years(self) -> list:
        """ years with available data for a dataset """
        pass

    @abc.abstractmethod
    def _list_content(self,
                      subdirs: Union[tuple, str] = '',
                      pattern: str = '*',
                      type: {'all', 'dir', 'file'} = 'all',
                      absolute: bool = False) -> list:
        """ List content of local or remote directory, use pattern to filter """
        pass

    @abc.abstractmethod
    def _join_path(self, subdirs: Union[str, tuple]):
        """ Append subdirs to root """
        pass

    @abc.abstractmethod
    def list_folders_days_for_year(self, year: int, as_int=True) -> list:
        """ List all days folders for a year of for a dataset """
        pass

    @abc.abstractmethod
    def list_folders_years(self, as_int=True) -> list:
        """ List all year folders for a dataset """
        pass

    def _filter_cont(self,
                     items: list,
                     filter_func) -> list:
        return [i for i in items if filter_func(i)]

    def get_first_last_item(self,
                            subdirs='',
                            pattern='*') -> (str, str):
        """
        Get first and last item (sorted by name) in the passed dir.

        Parameters
        ----------
        subdirs : Union[str, tuple], optional (default: '')
            Subdirs in root
        pattern: str, optional (default: '*')
            Regex pattern for items to search.

        Returns
        -------
        first : str
            First item in sorted content
        last : str
            Last item in sorted content
        """

        content = sorted(self._list_content(subdirs, pattern))
        return (content[0], content[-1]) if len(content) >= 1 else (None, None)

    def list_files(self,
                   subdirs='',
                   patterns='*') -> list:
        """
        Like _list_content but only lists files following the selected patterns.

        Parameters
        ----------
        subdirs : Union[str, tuple], optional (default: '')
            Subfolders in root where folders are searched.
        patterns : Union[str, tuple], optional (default: '*')
            Regex pattern to filter certain files.
        """
        files = []

        if isinstance(patterns, str):
            patterns = [patterns]

        for pattern in patterns:
            files += self._list_content(subdirs, pattern,
                                        type='file', absolute=False)

        return files

    def list_folders(self,
                     subdirs='',
                     pattern='*',
                     exclude=None) -> list:
        """
        Like _list_content but only lists folders, allows to exclude certain folders.

        Parameters
        ----------
        subdirs : Union[str, tuple], optional (default: '')
            Subfolders in root where folders are searched.
        pattern : str, optional (default: '*')
            Regex pattern to filter certain folders.
        exclude : list, optional (default: None)
            List of folders to exclude.
        """

        if exclude is None:
            exclude = []

        folders = self._list_content(subdirs, pattern=pattern, type='dir',
                                     absolute=False)

        return [f for f in folders if f not in exclude]

    def list_files_for_day(self, year, day, extension='.nc4') -> list:
        # Implements file extension as pattern to extract files for day of year.
        return self.list_files(subdirs=(str(year), f"{day:03}"), patterns=(f'*{extension}',))

    def parse_filename(self, filename='first'):
        """
        Derive dataset components from filename, such as name, version, etc.

        Parameters
        ----------
        filename : str, optional (default: None)
            Filename to parse, or
            - first to parse the first file (sorted by date)
            - last to parse the last file (sorted by date)


        Returns
        -------
        components: dict
            dataset, early Product indicator, datetime, version, file type
        """
        if filename.lower() == 'first':
            years = self.list_folders_years(as_int=False)
            days = self.list_folders_days_for_year(int(years[0]), as_int=False)
            filename = self.list_files(subdirs=(years[0], days[0]), patterns='*.nc4')[0]
        if filename.lower() == 'last':
            years = self.list_folders_years(as_int=False)
            days = self.list_folders_days_for_year(years[-1], as_int=False)
            filename = self.list_files(subdirs=(years[-1], days[-1]), patterns='*.nc4')[0]

        cont = parse(glob.data_fn_templ, filename).named

        try:
            cont['datetime'] = datetime.strptime(f"{cont['date']}.{cont['time']}",
                                                 glob.data_fn_templ_dt_format)
        except ValueError:
            cont['datetime'] = None

        return cont


class GldasLocal(DatasetConnection):
    """ Connection to local GLDAS data """

    def __init__(self, path):
        """
        Connects to a local folder where GLDAS image data is stored in annual / daily folders

        Parameters
        ----------
        path : str
            Local root path to the data directory that contains folders for each year.
        """

        if not os.path.exists(path):
            raise IOError(path)

        super(GldasLocal, self).__init__(path)

    def __repr__(self):
        return f"Local GDAS Data at {self.root}"

    def _list_content(self,
                     subdirs: Union[tuple, str] = '',
                     pattern: str = '*',
                     type: {'all', 'dir', 'file'} = 'all',
                     absolute:bool = False) -> list:
        # list content of local dir
        try:
            items = os.listdir(self._join_path(subdirs))
            items = fnmatch.filter(items, pattern)
            items = [os.path.join(self._join_path(subdirs), i) for i in items]

            if type.lower() == 'all':
                cont = items
            else:
                if type.lower() == 'dir':
                    cont = self._filter_cont(items, os.path.isdir)
                elif type.lower() == 'file':
                    cont = self._filter_cont(items, os.path.isfile)
                else:
                    raise NotImplementedError(f"Type not implemented: {type}")
        except FileNotFoundError:
            cont = []

        if not absolute:
            cont = [os.path.basename(c) for c in cont]

        return cont

    def _join_path(self, subdirs: Union[tuple, str]):
        if isinstance(subdirs, str):
            subdirs = (subdirs, )
        return os.path.join(self.root, *subdirs)

    def list_folders_years(self, as_int: bool=True) -> list:
        # Implements 4 digit pattern for list_folders() to extract folders for years in root.
        return [int(y) if as_int else y
                for y in self.list_folders(pattern="[0-9][0-9][0-9][0-9]")]

    def list_folders_days_for_year(self, year: int, as_int: bool=True) -> list:
        # Implements 3 digit pattern for list_folders() in folder for year to extract folders for days.
        return [int(d) if as_int else d
                for d in self.list_folders(subdirs=(str(year),), pattern="[0-9][0-9][0-9]")]


class GldasRemote(DatasetConnection):
    """ Connection to remote GLDAS data """

    def __init__(self,
                 dataset: str = None):

        self.url = "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/"
        self.dataset = dataset
        self._datasets = self.list_folders(subdirs='', pattern='GLDAS*')

    @property
    def root(self):
        return self.url if self.dataset is None else urljoin(self.url, self.dataset)

    def __repr__(self):
        if self.dataset is None:
            return f"Available datasets {self._datasets}. Use GldasRemote.connect()."
        else:
            return f"Connected to {self.dataset} @ {self.root}"

    def _join_path(self, subdirs: Union[tuple, str]) -> str:
        if isinstance(subdirs, str):
            subdirs = (subdirs, )
        #return reduce(urljoin(self.root), subdirs)
        return '/'.join((self.root,) + subdirs)

    def connect(self, dataset:str):
        "Connect to a dataset that online at url"
        if dataset not in self._datasets:
            raise ValueError(f"{dataset} not available at {self.url}. "
                             f"Select one of: {self._datasets}")
        self.dataset = dataset

    def _list_content(self,
                      subdirs: Union[tuple, str] = '',
                      pattern='*',
                      type='all',
                      absolute: bool = False) -> list:
        """
        List content from connection.

        Parameters
        ----------
        subdirs : List of subdirs
        pattern : naming pattern to filter files/folders
        type : 'all', 'dir' or 'file'
        absolute : Whether to return only names or full paths
        """
        try:
            _ignore = ['Parent Directory', 'NaN']
            table = pd.read_html(self._join_path(subdirs))[0]
            items = []
            for v in table['Name'].dropna().values:
                if v not in _ignore and isinstance(v, str):
                    items.append(v)
            items = fnmatch.filter(items, pattern)
            items = [urljoin(self.root, i) for i in items]

            if type.lower() == 'all':
                cont = items
            elif type.lower() == 'dir':
                cont = self._filter_cont(items, lambda x: x.endswith('/'))
            elif type.lower() == 'file':
                cont = self._filter_cont(items, lambda x: not x.endswith('/'))
            else:
                raise NotImplementedError(f"Type not implemented: {type}")
        except urllib.error.HTTPError:
            cont = []

        if not absolute:
            cont = [os.path.basename(os.path.abspath(urlparse(c).path)) for c in cont]

        return cont

    # @needs_dataset
    # def list_files_for_day(self, year, day, extension='.nc4'):
    #     subdirs = (self.dataset, str(year), f"{day:03}")
    #     return self.list_files(subdirs=subdirs, patterns=(f'*{extension}',))

    def list_folders_years(self, as_int: bool=True) -> list:
        # Implements 4 digit + / pattern for list_folders() to extract folders for years in remote.
        return [int(y) if as_int else y
                for y in self.list_folders(pattern="[0-9][0-9][0-9][0-9]/")]

    def list_folders_days_for_year(self, year:int, as_int: bool=True) -> list:
        # Implements 3 digit + / pattern for list_folders() in folder for year to extract folders for days.
        return [int(d) if as_int else d
                for d in self.list_folders(subdirs=(str(year),), pattern="[0-9][0-9][0-9]/")]

if __name__ == '__main__':
    remote = GldasRemote()
    remote.connect('GLDAS_NOAH025_3H.2.1')
    first, last = remote.get_first_last_item(('2015', '001'), pattern='*.nc')
    folders = remote.list_folders(exclude=('doc',))
    years = remote.list_folders_years()
    days = remote.list_folders_days_for_year(2015)
    days = remote.list_folders_days_for_year(2022)
    files = remote.list_files_for_day(2015, 1, '.nc4')


    local = GldasLocal("/home/wolfgang/code/gldas/tests/test-data/GLDAS_NOAH_image_data")
    first, last = local.get_first_last_item(('2015', '001'))
    folders = local.list_folders(exclude=('doc',))
    years = local.list_folders_years()
    days = local.list_folders_days_for_year(2015)
    days = local.list_folders_days_for_year(2022)
    files = local.list_files_for_day(2015, 1, '.nc4')
