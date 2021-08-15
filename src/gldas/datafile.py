
"""
df = pd.read_xml("/home/wolfgang/data-read/temp/2015/001/GLDAS_NOAH025_3H.A20150101.0000.021.nc4.xml", xpath='.//MeasuredParameter')
df = pd.read_xml("/home/wolfgang/data-read/temp/2015/001/GLDAS_NOAH025_3H.A20150101.0000.021.nc4.xml", xpath='.//BoundingRectangle')
df = pd.read_xml("/home/wolfgang/data-read/temp/2015/001/GLDAS_NOAH025_3H.A20150101.0000.021.nc4.xml", xpath='.//RangeDateTime')
df = pd.read_xml("/home/wolfgang/data-read/temp/2015/001/GLDAS_NOAH025_3H.A20150101.0000.021.nc4.xml", xpath='.//DataGranule')
df = pd.read_xml("/home/wolfgang/data-read/temp/2015/001/GLDAS_NOAH025_3H.A20150101.0000.021.nc4.xml", xpath='.//CollectionMetaData')

https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/GLDAS_NOAH025_3H.xml



"""

from dataclasses import dataclass
