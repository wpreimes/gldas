# The MIT License (MIT)
#
# Copyright (c) 2018, TU Wien
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module managing download of NOAH GLDAS data.
"""

import os
import sys
import glob
import argparse
from functools import partial

from gldas.connections import GldasRemote, GldasLocal

from trollsift.parser import validate, parse, globify
from datetime import datetime
from datedown.interface import mkdate
from datedown.dates import daily
from datedown.urlcreator import create_dt_url
from datedown.fname_creator import create_dt_fpath
from datedown.interface import download_by_dt
from datedown.down import download



def get_time_range(dataset=None):
    """
    Get NOAH GLDAS start dates for one or all products.

    Parameters
    ----------
    dataset : str, optional (default: None)
        Product name, if None is passed, all are checked

    Returns
    -------
    prod_time_range : dict[str, tuple]
        GLDAS Product and time ranges-
    """
    prod_time_range = dict()
    remote = GldasRemote()
    available_datasets = remote.available_datasets

    if dataset is not None:
        if dataset not in available_datasets:
            raise ValueError(f"Passed dataset {dataset} is not one of "
                             f"the available datasets {available_datasets}")
        available_datasets = [dataset]

    for d in sorted(available_datasets):
        remote.connect(d)

        prod_time_range[d] = remote.get_first_last_item()

        # if dataset is None:

    dt_dict = {
        "GLDAS_Noah_v20_025": datetime(1948, 1, 1, 3),
        "GLDAS_Noah_v21_025": datetime(2000, 1, 1, 3),
        "GLDAS_Noah_v21_025_EP": datetime(2000, 1, 1, 3),
    }

    return dt_dict[product]


def parse_args(args):
    """
    Parse command line parameters for recursive download.

    Parameters
    ----------
    args : list of str
        Command line parameters as list of strings.

    Returns
    -------
    args : argparse.Namespace
        Command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download GLDAS data.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "localroot",
        help="Root of local filesystem where" "the data is stored.",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=mkdate,
        help=(
            "Startdate. Either in format YYYY-MM-DD or "
            "YYYY-MM-DDTHH:MM. If not given then the target"
            "folder is scanned for a start date. If no data"
            "is found there then the first available date "
            "of the product is used."
        ),
    )

    parser.add_argument(
        "-e",
        "--end",
        type=mkdate,
        help=(
            "Enddate. Either in format YYYY-MM-DD or "
            "YYYY-MM-DDTHH:MM. If not given then the "
            "current date is used."
        ),
    )

    help_string = "\n".join(
        [
            "GLDAS product to download.",
            "GLDAS_Noah_v20_025 available from {} to 2014-12-31",
            "GLDAS_Noah_v21_025 available from {}",
            "GLDAS_Noah_v21_025_EP available after GLDAS_Noah_v21_025",
        ]
    )

    help_string = help_string.format(
        get_gldas_start_date("GLDAS_Noah_v20_025"),
        get_gldas_start_date("GLDAS_Noah_v21_025"),
    )

    parser.add_argument(
        "--product",
        choices=[
            "GLDAS_Noah_v20_025",
            "GLDAS_Noah_v21_025",
            "GLDAS_Noah_v21_025_EP",
        ],
        default="GLDAS_Noah_v21_025",
        help=help_string,
    )

    parser.add_argument("--username", help="Username to use for download.")

    parser.add_argument("--password", help="password to use for download.")

    parser.add_argument(
        "--n_proc",
        default=1,
        type=int,
        help="Number of parallel processes to use for" "downloading.",
    )

    args = parser.parse_args(args)
    # set defaults that can not be handled by argparse

    # Compare versions to prevent mixing data sets
    version, first, last = gldas_folder_get_version_first_last(args.localroot)
    if args.product and version and (args.product != version):
        raise Exception(
            "Error: Found products of different version ({}) "
            "in {}. Abort download!".format(version, args.localroot)
        )

    if args.start is None or args.end is None:
        if not args.product:
            args.product = version
        if args.start is None:
            if last is None:
                if args.product:
                    args.start = get_gldas_start_date(args.product)
                else:
                    # In case of no indication if version, use GLDAS Noah 2.0
                    # start time, because it has the longest time span
                    args.start = get_gldas_start_date("GLDAS_Noah_v20_025")
            else:
                args.start = last
        if args.end is None:
            args.end = datetime.now()

    prod_urls = {
        "GLDAS_Noah_v20_025": {
            "root": "hydro1.gesdisc.eosdis.nasa.gov",
            "dirs": ["data", "GLDAS", "GLDAS_NOAH025_3H.2.0", "%Y", "%j"],
        },
        "GLDAS_Noah_v21_025": {
            "root": "hydro1.gesdisc.eosdis.nasa.gov",
            "dirs": ["data", "GLDAS", "GLDAS_NOAH025_3H.2.1", "%Y", "%j"],
        },
        "GLDAS_Noah_v21_025_EP": {
            "root": "hydro1.gesdisc.eosdis.nasa.gov",
            "dirs": ["data", "GLDAS", "GLDAS_NOAH025_3H_EP.2.1", "%Y", "%j"],
        },
    }

    args.urlroot = prod_urls[args.product]["root"]
    args.urlsubdirs = prod_urls[args.product]["dirs"]
    args.localsubdirs = ["%Y", "%j"]

    print(
        "Downloading data from {} to {} "
        "into folder {}.".format(
            args.start.isoformat(), args.end.isoformat(), args.localroot
        )
    )
    return args


def main(args):
    """
    Main routine used for command line interface.

    Parameters
    ----------
    args : list of str
        Command line arguments.
    """
    args = parse_args(args)

    dts = list(daily(args.start, args.end))
    url_create_fn = partial(
        create_dt_url, root=args.urlroot, fname="", subdirs=args.urlsubdirs
    )
    fname_create_fn = partial(
        create_dt_fpath,
        root=args.localroot,
        fname="",
        subdirs=args.localsubdirs,
    )

    down_func = partial(
        download,
        num_proc=args.n_proc,
        username=args.username,
        password="'" + args.password + "'",
        recursive=True,
        filetypes=["nc4", "nc4.xml"],
    )
    download_by_dt(
        dts, url_create_fn, fname_create_fn, down_func, recursive=True
    )


def run():
    main(sys.argv[1:])
