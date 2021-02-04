from pathlib import Path

import cudf
import dask_cudf
import geopandas as gpd
import pandas as pd


class Config:
    DATA_PATH = Path("data/")
    RAW_DATA = DATA_PATH / "raw"
    PROCESSED_DATA = DATA_PATH / "processed"
    OSM_GRAPH = PROCESSED_DATA / "osm"

    NODE_COLS = ['id', 'easting', 'northing']
    EDGE_COLS = ['u', 'v', 'length', 'maxspeed']


def clean_postcodes(path: Path) -> cudf.DataFrame:
    pc_path = Path(path).glob("**/*.csv")
    postcodes = dask_cudf.read_csv(
        list(pc_path),
        header=None,
        usecols=[0, 2, 3]
    )
    postcodes = postcodes.rename(  # type: ignore
        columns={"0": "postcode",
                 "2": "easting",
                 "3": "northing"}
    )
    postcodes = postcodes[(postcodes.easting != 0) & (postcodes.northing != 0)]
    postcodes = postcodes.drop_duplicates(subset=['easting', 'northing'])
    return postcodes.compute()


def clean_retail_centres(path: Path) -> pd.DataFrame:
    path = Config.RAW_DATA / "retailcentrecentroids.gpkg"
    retail = gpd.read_file(path)
    retail['easting'], retail['northing'] = retail.geometry.x, retail.geometry.y
    retail = pd.DataFrame(retail.drop('geometry', axis=1))
    return retail
