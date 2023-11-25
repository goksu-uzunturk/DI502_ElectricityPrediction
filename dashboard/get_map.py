import geopandas as gpd
import pandas as pd
from shapely import wkt


ISO_MAP_IDS = {
    56669: 'MISO',
    14725: 'PJM',
    2775: 'CAISO',
    13434: 'ISONE',
    13501: 'NYISO'
}

