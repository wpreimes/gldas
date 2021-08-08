import urllib.error
import numpy as np
import pandas as pd
import fnmatch
import abc
import os
import re
import urllib.request as request
from urllib.parse import urlparse, urljoin
from functools import wraps, reduce
from parse import parse
from datetime import datetime, timedelta
import gldas.const as globals
import time
from typing import Callable, Optional, Union, Tuple
from typing_extensions import Literal
from warnings import warn
import requests
import subprocess
import getpass
from dataclasses import dataclass
from pathlib import Path
import glob

"""

todo:
[] - pydap access still not working
[] - parallel processes to filedown to parallelise all download functions



import xarray as xr
from pydap.cas.urs import setup_session
ds_url = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_CLSM10_M.2.1/2015/GLDAS_CLSM10_M.A201503.021.nc4'
session = setup_session('wpreimes', 'ThisTestPWD1', check_url=ds_url)
store = xr.backends.PydapDataStore.open(ds_url, session=session)
ds = xr.open_dataset(store)

"""
class SessionWithHeaderRedirection(requests.Session):
    # see: https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    AUTH_HOST = globals.AUTH_HOST
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):

        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) \
                and redirect_parsed.hostname != self.AUTH_HOST \
                and original_parsed.hostname != self.AUTH_HOST:

                del headers['Authorization']


def needs_dataset(func: Callable):
    @wraps(func)
    def _impl(self, *args, **kwargs):
        if self.dataset is None:
           raise IOError("Not conncected to any dataset")
        return func(self, *args, **kwargs)
    return _impl


@dataclass
class DatasetConnection:
    """ 
    GLDAS data collection (remote or local)
    """

    is_remote: bool

    @abc.abstractclassmethod
    def _list_content(self,
                      subdirs: Union[tuple, str] = '',
                      ignore: Optional[list] = None,
                      patterns: Optional[Union[str, tuple]] = '*',
                      type: Optional[Literal['all', 'dir', 'file']] = 'all',
                      absolute: Optional[bool] = False,
                      ) -> dict:
        """
        Create a collection of files available for a data set.
        Return a dictionary containing a field for Filename, modification date,
        file size.

        Parameters
        ----------
        subdirs: str or tuple, optional (default: '')
            folder or list of folders for which the content is returned.
        ignore: str or list[str], optional (default: None)
            List of names that should be ignored when listing contents
        patterns: str or tuple, optional (default: *)
            Filename/folder patterns, all elemtns are matched via fnmatch.
        type: 'all' or 'dir' or 'file', optional (default: 'all')
            Whether to consider file and directory names (all) or only file names (file)
            or only folder names (dir).
        absolute: bool, optional (default: False)
            Whether to return the full path or the file/folder name only.

        Returns
        -------
        cont: dict
            dictionary that contains file/folder names, and properties.
        """
        pass

    @abc.abstractproperty
    def root(self) -> str:
        # Root path or URL to remote or local data set
        pass

    @abc.abstractclassmethod
    def _join_path(self, subdirs: Union[str, Tuple[str]]) -> str:
        # Must implement a way to join subdirs to URL or local path
        pass

    @abc.abstractstaticmethod
    def _ffilter(path: str,
                 ignore: Optional[Union[list, str]] = None,
                 patterns: Optional[Union[str, tuple]] = '*',
                 type: Optional[Literal['all', 'dir', 'file']] = 'all') \
            -> np.ndarray:
        """ Filter function to differentiate between files and folders. """
        pass

    def list_years(self, as_int: Optional[bool]=False) -> list:
        """
        Implements 4 digit pattern for list_folders() to extract yearly folders
        for the connected dataset.

        Parameters
        ----------
        as_int : bool, optional (default: False)
            Convert the subdirs to integers instead of return the strings.

        Returns
        -------
        years : list
            Years that contain data for the connected dataset.
        """

        pattern = "[0-9][0-9][0-9][0-9]"

        if self.is_remote:
            patterns = f"{pattern}/"

        return [int(y) if as_int else y
                for y in self.list_folders(patterns=pattern)]

    def _infos_from_files(self) -> (str, str, str, str, datetime, datetime):
        # detect subdir structure and fn template
        # this is done after connecting to a dataset (remote or local)

        years = self.list_years(as_int=False)

        first_year, last_year = years[0], years[-1]
        first_folder_subdirs = sorted(self.list_folders(first_year))
        last_folder_subdirs = sorted(self.list_folders(last_year))

        if len(first_folder_subdirs) == len(last_folder_subdirs) == 0:
            subdir_pattern = None  # all files for one year in one dir
            subdir_templ = ('%Y', )
            subdirs_first = (first_year,)
            subdirs_last = (last_year,)
        else:
            if len(first_folder_subdirs[0]) == 2 or len(last_folder_subdirs[0]) == 2:
                # monthly subdirs
                subdir_pattern = "[0-9][0-9]/" if self.is_remote else "[0-9][0-9]"
                subdir_templ = ('%Y', '%m')
            else:
                # doy subdirs
                subdir_pattern = "[0-9][0-9][0-9]/" if self.is_remote else "[0-9][0-9][0-9]"
                subdir_templ = ('%Y', '%j')
            subdirs_first = (first_year, first_folder_subdirs[0])
            subdirs_last = (last_year, last_folder_subdirs[-1])

        first_file = self.list_files(subdirs=subdirs_first, patterns='*.nc4')[0]
        last_file = self.list_files(subdirs=subdirs_last, patterns='*.nc4')[-1]

        cont, fntempl, dttempl = self.parse_filename(first_file)

        p, fn_templ, dttempl = self.parse_filename(first_file)
        start_date = p['datetime']
        p, _, _ = self.parse_filename(last_file)
        end_date = p['datetime']

        return subdir_pattern, subdir_templ, fntempl, dttempl, start_date, end_date

    def get_first_last_item(self,
                            subdirs='',
                            patterns='*',
                            **kwargs) -> (str, str):
        """
        Get first and last item (sorted by name) in the passed dir.

        Parameters
        ----------
        subdirs : Union[str, tuple], optional (default: '')
            Subdirs in root
        patterns: str or tuple, optional (default: '*')
            Regex patterns for items to search.
        kwargs : additional kwargs are passed to _list_content()

        Returns
        -------
        first : str
            First item in sorted content
        last : str
            Last item in sorted content
        """

        content = sorted(self._list_content(
            subdirs, ignore=None, patterns=patterns, **kwargs)['Name'])

        return (content[0], content[-1]) if len(content) >= 1 else (None, None)

    def list_files(self,
                   subdirs='',
                   recursive=False,
                   patterns='*',
                   **kwargs) -> list:
        """
        Like _list_content but only lists files following the selected patterns.

        Parameters
        ----------
        subdirs : Union[str, tuple], optional (default: '')
            Subfolders in root where folders are searched.
        recursive: bool, optional (default: False)
            Look for files also in subdirs
        patterns : Union[str, tuple], optional (default: '*')
            Regex patterns to filter certain FILES.
        kwargs : additional kwargs are passed to _list_content()
        """
        files = []

        if isinstance(subdirs, str):
            subdirs = (subdirs,)

        if isinstance(patterns, str):
            patterns = [patterns]

        if not recursive:
            folders = tuple()
        else:
            folders = self.list_folders(subdirs, patterns='*')

        if len(folders) == 0:
            cont = self._list_content(subdirs,
                                      patterns=patterns,
                                      type='file',
                                      **kwargs)
            files += cont['Name']
        else:
            for folder in folders:
                print(folder)
                files += self.list_files(tuple([*subdirs, folder]),
                                         recursive=recursive,
                                         patterns=patterns,
                                         **kwargs)
        return files

    def list_folders(self,
                     subdirs='',
                     patterns='*',
                     **kwargs) -> list:
        """
        Like _list_content but only lists folders, allows to exclude certain folders.

        Parameters
        ----------
        subdirs: Union[str, tuple], optional (default: '')
            Subfolders in root where folders are searched.
        patterns: str or tuple, optional (default: '*')
            Regex patterns to filter certain folders.
        kwargs: additional kwargs are passed to _list_content()
        """
        if isinstance(subdirs, str):
            subdirs = (subdirs,)

        subdirs = tuple([str(s) for s in subdirs])

        folders = self._list_content(subdirs,
                                     patterns=patterns,
                                     type='dir',
                                     **kwargs)['Name']

        return folders

    def list_subdirs_for_year(self,
                              year: Union[str, int],
                              as_int: Optional[bool]=False) -> list:
        """
        List the subdirs of a year for the currely connected dataset.

        Parameters
        ----------
        year : int or str
            Year to list subfolders for
        as_int : bool, optional (default: False)
            Return the subdirs as integers instead of strings

        Returns
        -------
        subdirs : list
            List of subdirs for the selected year
        """
        if self.subdir_pattern is None:
            return list()
        else:
            return [int(d) if as_int else d
                    for d in self.list_folders(subdirs=(str(year),),
                                               patterns=self.subdir_pattern)]

    def list_files_for_subdir(self,
                              year: int,
                              subdir: Optional[str] = None,
                              extension: Optional[str] = '.nc4') -> list:
        """
        Implements file extension as patterns to extract files from subdir

        Parameters
        ----------
        year : int
            Year to extract files for
        subdir : str, optional (default: None)
            Subdir to extract files for, e.g. 001 for DOY structure, or 12 for month
            structure or None if files are location in the year folder.

        Returns
        -------
        files : list
            List of files in the subdir for year
        """
        if subdir is None:
            subdirs = (str(year),)
        else:
            subdirs = (str(year), subdir)

        if self.subdir_pattern is None:
            if subdir is not None:
                raise ValueError("No subdirs for year for the currently "
                                 "active dataset.")

        return self.list_files(subdirs=subdirs, patterns=f'*{extension}')

    def parse_filename(self, filename):
        """
        Derive dataset info from filename.
        Elements such as name, version, etc.

        Parameters
        ----------
        filename : str
            Filename to parse

        Returns
        -------
        cont: dict
            dataset, early Product indicator, datetime, version, file type
        fntempl : str
            Filename template.
        dttempl: str
            Datetime template used in filename
        """
        for fntempl in globals.fn_templates:
            cont = parse(fntempl, filename)
            if cont:
                break

        cont = cont.named

        if len(cont['date']) == 6:
            dttempl = "%Y%m"
            cont['datetime'] = datetime.strptime(f"{cont['date']}", dttempl)  # datetime needs a day
        elif 'time' in cont.keys():
            dttempl =  "%Y%m%d.%H%M"
            cont['datetime'] = datetime.strptime(f"{cont['date']}.{cont['time']}", dttempl)
            fntempl = fntempl.replace('.{time}', '')
        else:
            dttempl = "%Y%m%d"
            cont['datetime'] = datetime.strptime(f"{cont['date']}", dttempl)

        fntempl = fntempl.format(shortname=cont['shortname'], date=dttempl,
                                 vers_id=cont['vers_id'], ext=cont['ext'])

        return cont, fntempl, dttempl


class GldasRemote(DatasetConnection):
    """ Connection to remote GLDAS data """

    # Sets:
    dataset : str
    subdir_pattern : str
    subdir_templ : str
    fntempl : str
    time_range : Tuple[datetime, datetime]
    session : SessionWithHeaderRedirection

    def __init__(self,
                 dataset: Optional[str]=None,
                 username: Optional[str]=None,
                 password: Optional[str]=None):
        """
        Connection to remote dataset folder.

        Parameters
        ----------
        dataset : str, optional (default: None)
            Name of an available dataset to connect to. If None
            is passed, connect() must be used to connect to connect
            to a dataset before further use.
        username: str, Optional (default: None)
            Username used to log in at 'urs.earthdata.nasa.gov'
            If None is passed, no data can be downloaded.
        password: str, Optional (default: None)
            Password used to log in at 'urs.earthdata.nasa.gov'.
            If None is passed, no data can be downloaded.
        """
        self.url = globals.gldas_url
        self.available_datasets = self.list_folders(subdirs='', patterns='GLDAS*')

        super(GldasRemote, self).__init__(is_remote=True)

        if dataset is None:
            self.dataset = None  # connect later
        else:
            # TODO: if user/pwd is none use the env variables, if they are not set, show descriptive error here.
            self.connect(dataset, username, password)  # connect directly


    def __repr__(self) -> str:
        if self.dataset is None:
            ds = "\n  ".join(self.available_datasets)
            return f"Not connected to any available remote dataset. " \
                   f"Use `GldasRemote.connect()` with one of:\n" \
                   f"  {ds}"
        else:
            return f"Connected to `{self.dataset}` " \
                   f"available from {str(self.time_range[0])} to " \
                   f"{str(self.time_range[1])} at {self.root}"

    @property
    def root(self) -> str:
        if not hasattr(self, 'dataset') or self.dataset is None:
            return self.url
        else:
            return urljoin(self.url, self.dataset)

    def connect(self,
                dataset:str,
                username: Optional[str]=None,
                password: Optional[str]=None):
        """
        Connect to one of the datasets that is listed in available_datasets.
        And detect subdir structure and fn patterns

        Parameters
        ----------
        dataset : str
            Name of the dataset to connect to.
        username: str, Optional (default: None)
            Username used to log in at 'urs.earthdata.nasa.gov'
            If None is passed, no data can be downloaded.
        password: str, Optional (default: None)
            Password used to log in at 'urs.earthdata.nasa.gov'.
            If None is passed, no data can be downloaded.
        """

        if dataset not in self.available_datasets:
            raise ValueError(f"{dataset} not available at {self.url}. "
                             f"Select one of: {self.available_datasets}")
        self.dataset = dataset

        self.subdir_pattern, self.subdir_templ, self.fntempl, \
        dttempl, start_date, end_date = self._infos_from_files()
        self.time_range = (start_date, end_date)

        if password is None or username is None:
            password = os.environ['GLDAS_PWD']
            username = os.environ['GLDAS_USER']

        if password is None or username is None:
            self.session = None
        else:
            self.session = SessionWithHeaderRedirection(username, password)

    def _join_path(self, subdirs: Union[tuple, str]) -> str:
        if isinstance(subdirs, str):
            subdirs = (subdirs, )
        return '/'.join((self.root,) + subdirs)

    @staticmethod
    def _ffilter(path: str,
                 ignore: Optional[list] = None,
                 patterns: Optional[Union[str, tuple]] = ('*',),
                 type: Optional[Literal['all', 'dir', 'file']] = 'all') \
            -> np.ndarray:
        """ Filter function to differentiate between files and folders. """
        if ignore is None:
            ignore = []
        elif isinstance(ignore, str):
            ignore = [ignore]
        if isinstance(patterns, str):
            patterns = [patterns]
            
        ignore += ['Parent Directory', 'NaN'] # ALWAYS ignored

        flags = [path not in ignore,
                 isinstance(path, str),
                 np.any([fnmatch.fnmatch(path, p) for p in patterns])]

        if type.lower() == 'all':
            pass
        elif type.lower() == 'dir':
            flags.append(path.endswith('/'))
        elif type.lower() == 'file':
            flags.append(not path.endswith('/'))
        else:
            raise NotImplementedError(f"Type not implemented: {type}")

        return np.all(flags)
        
    def _list_content(
            self,
            subdirs: Union[tuple, str] = '',
            absolute: Optional[bool]=False,
            **filter_kwargs
            ) -> dict:
        """
        List content from remote connection.

        Parameters
        ----------
        subdirs : List of subdirs
        ignore: List of names to ignore
        patterns : naming patterns to filter files/folders
        type : 'all', 'dir' or 'file'
        absolute : Whether to return only names or full paths
        """
        # try:
        table = pd.read_html(self._join_path(subdirs))[0]
        columns = ['Name']
        columns.append('Last modified')
        columns.append('Size')

        table.dropna(subset=('Name',), inplace=True)

        flags = np.array([self._ffilter(path=path, **filter_kwargs)
                          for path in table['Name'].values])
        table = table.loc[flags, columns]
        cont = table.to_dict(orient='list')
        # except urllib.error.HTTPError:
        #     cont = {}

        names = []
        for n in cont['Name'].copy():
            if absolute:
                n = self._join_path((*subdirs, n))
            if n.endswith('/'):
                n = n[:-1]
            names.append(n)

        cont['Name'] = names

        return cont

    def list_years(self, as_int: Optional[bool]=False) -> list:
        """
        Implements 4 digit patterns for list_folders() to extract yearly folders
        for the connected dataset.

        Parameters
        ----------
        as_int : bool, optional (default: False)
            Convert the subdirs to integers instead of return the strings.

        Returns
        -------
        years : list
            Years that contain data for the connected dataset.
        """

        return [int(y) if as_int else y
                for y in self.list_folders(patterns="[0-9][0-9][0-9][0-9]/")]

    def url4date(self, dt: datetime) -> str:
        """
        Create the URL to a file for the current dataset for a certain date
        """
        subdirs = [dt.strftime(e) for e in self.subdir_templ]

        if len(subdirs) == 1:
            subdirs = (subdirs[0], )
        else:
            subdirs = tuple(subdirs)

        fn = dt.strftime(self.fntempl)

        return f"{self._join_path(subdirs)}/{fn}"


    def filedown(self,
                 urls: Union[str, list],
                 create_subdirs=True,
                 local_root: Optional[str] = None):
        """
        Download remote netcd file via its URL.

        Parameters
        ----------
        url: str or list
            File URL(s)
        create_subdirs: bool, optional (default: True)
            Create subfolder structure as for the remote dataset and place
            downloaded file in the respective directory.
        local_root : str, optional (default: None)
            Local directory under which the data is stored.

        Returns
        -------

        """
        if self.session is None:
            raise ValueError('No username/password found')

        for url in urls:
            remote_subdirs = url.replace(self.root, '').split('/')
            if not create_subdirs:
                filepath = remote_subdirs[-1]
            else:
                filepath = os.path.join(*remote_subdirs)

            path_local = os.path.join(local_root, filepath)
            os.makedirs(os.path.dirname(path_local), exist_ok=True)
            try:
                response = self.session.get(url, stream=True)
                response.raise_for_status()
                print(url)
                with open(path_local, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        fd.write(chunk)
            except requests.exceptions.HTTPError as e:
                print(e)

    def datedown(self,
                 dt: datetime,
                 local_root: str,
                 xml: Optional[bool]=False):
        """
        Download a specific netcdffile for the current dataset by its date

        Parameters
        ----------
        dt: datetime
            Date time of the file to download if it exists
        local_root: str
            Local path where the downloaded file is stored.
        xml: bool, Optional (default: False)
            Also download the accoring xml file for the nc file.
        """

        self.filedown(self.url4date(dt), local_path, xml)


    def dirdown(self,
                local_root: str,
                subdirs: Optional[Union[str, tuple]]=None,
                recursive: Optional[bool]=True,
                patterns=('*.nc4', '*.nc4.xml'),
                **kwargs):
        """
        Download ALL netcdf files in a subdir

        Parameters
        ----------
        local_root: str
            Local root path where files are stored (must already exist)
        subdirs: tuple or str, optional (default: None)
            Subdir of a dataset to download, typically something like (2000, 001)
            If no subdir is passed, the dataset root is used, ie. we look through
            the whole data set, which can take some time.
        xml: bool, Optional (default: False)
            Also download the accoring xml file for the nc file.
        recursive : bool, Optional (default: True)
            Downloads all netcdf file in the passed subdir recursively.
        patterns: str or tuple(str), optional (default: ('*.nc4', '*.nc4.xml'))
            Filename patterns of files to download
        """

        urls = self.list_files(subdirs,
                               recursive=recursive,
                               patterns=patterns,
                               absolute=True,
                               **kwargs)
        self.filedown(urls, create_subdirs=True, local_root=local_root)



class GldasLocal(DatasetConnection):
    """
    Connection to local GLDAS data
    """

    path: str

    def __init__(self,
                 path:str):
        """
        Connects to a local folder where GLDAS image data is stored in annual
        / doy folders

        Parameters
        ----------
        path : str
            Local root path to the data directory that contains folders for
            each year.
        """

        if not os.path.exists(path):
            raise IOError(path)

        super().__init__(is_remote=False)

        self.path = path

        self.subdir_pattern, self.subdir_templ, self.fntempl, \
        dttempl, start_date, end_date = self._infos_from_files()

    @property
    def root(self) -> str:
        return self.path

    def _join_path(self, subdirs: Union[tuple, str]):
        if isinstance(subdirs, str):
            subdirs = (subdirs, )
        return os.path.join(self.root, *subdirs)

    @staticmethod
    def _ffilter(path: str,
                 ignore: Optional[Union[list, str]] = None,
                 patterns: Optional[Union[str, tuple]] = '*',
                 type: Optional[Literal['all', 'dir', 'file']] = 'all') \
            -> np.ndarray:
        """ Filter function to differentiate between files and folders. """
        if ignore is None:
            ignore = []
        elif isinstance(ignore, str):
            ignore = [ignore]

        if isinstance(patterns, str):
            patterns = (patterns,)

        name = os.path.basename(path)

        flags = [name not in ignore,
                 isinstance(name, str),
                 np.any([fnmatch.fnmatch(name, p) for p in patterns])]

        if type.lower() == 'all':
            pass
        elif type.lower() == 'dir':
            flags.append(os.path.isdir(path))
        elif type.lower() == 'file':
            flags.append(os.path.isfile(path))
        else:
            raise NotImplementedError(f"Type not implemented: {type}")

        return np.all(flags)

    def _list_content(
            self,
            subdirs: Union[tuple, str] = '',
            absolute: Optional[bool] = False,
            **filter_kwargs
            ) -> dict:
        """
        List content from local connection.

        Parameters
        ----------
        subdirs : List of subdirs
        ignore: List of names to ignore, default None
        patterns : naming patterns to filter files/folders, default '*' for all files
        type : 'all', 'dir' or 'file', default 'all'
        absolute : Whether to return only names or full paths
        """


        # get all files and folders in subdir
        glob_pattern = self._join_path(tuple([*subdirs, '*']))
        paths = np.array([f for f in glob.glob(glob_pattern)])

        if len(paths) > 0:
            flags = np.array([self._ffilter(path=path,
                                            **filter_kwargs)
                              for path in paths])
        else:
            flags = slice(None, None, None)

        cont = {'sizes': [], 'Last modified': [], 'Name': []}

        for p in paths[flags]:
            p = Path(p)
            cont['Last modified'].append(datetime.fromtimestamp(p.stat().st_mtime))
            cont['sizes'].append(os.path.getsize(p))
            cont['Name'].append(p.name if not absolute else str(p))

        return cont

    def __repr__(self):
        return f"Local GLDAS Data: {self.root} \n "
            # find out what dataset is in root from filenames?
            #    f"{super().__repr__().rjust(3, ' ')}"


if __name__ == '__main__':
    import xarray as xr


    remote = GldasRemote('GLDAS_CLSM10_3H_EP.2.1')
    print(remote)
    print(remote.session)

    urls = remote.dirdown(local_root="/home/wolfgang/data-write/temp/gldas/",
                          subdirs=('2021','121'), recursive=True)

    #
    # store = xr.backends.PydapDataStore.open(urls[0],
    #                                         session=remote.session)
    # ds = xr.open_dataset(store)
    #
    # xr.open_dataset(urls[0], backend_kwargs=dict(session=remote.session))
    #

    # remote.datedown(datetime(2003,5,3,9), '/home/wolfgang/data-write/temp/gldas/', xml=True)
    # remote.dirdown(subdirs=None, recursive=True, local_root='/home/wolfgang/data-write/temp/gldas/',
    #                xml=False)
    # print(remote)
    # print(remote.list_folders()) # first level)
    # print(remote.list_years()) # only years
    # print(remote.list_subdirs_for_year(2005))
    # print(remote.list_files_for_subdir('2005', '01'))
    #
    # remote.datedown(datetime(2005,1,1),
    #                 path='/home/wolfgang/data-write/temp/gldas/',
    #                 xml=True)


    # remote.parse_filename()
    # first, last = remote.get_first_last_item(('2015', '001'), patterns='*.nc4')
    # folders = remote.list_folders(ignore=('doc',))
    # years = remote.list_subdirs_in_dataset()
    # days = remote.list_subdirs_in_dir(2015)
    # days = remote.list_subdirs_in_dir(2022)
    # files = remote.list_files_for_day(2015, 1, '.nc4')
    #
    #
    # local = GldasLocal("/home/wolfgang/code/gldas/tests/test-data/GLDAS_NOAH_image_data")
    # first, last = local.get_first_last_item(('2015', '001'))
    # folders = local.list_folders(ignore=('doc',))
    # years = local.list_subdirs_in_dataset()
    # days = local.list_subdirs_in_dir(2015)
    # days = local.list_subdirs_in_dir(2022)
    # files = local.list_files_for_day(2015, 1, '.nc4')
    #

