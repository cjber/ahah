import os
from pathlib import Path
import sys

import cudf
import geopandas as gpd
import pandas as pd


class Config:
    """Data paths"""

    DATA_PATH = Path("data/")
    RAW_DATA = DATA_PATH / "raw"
    PROCESSED_DATA = DATA_PATH / "processed"
    OUT_DATA = DATA_PATH / "out"
    OSM_GRAPH = PROCESSED_DATA / "osm"

    """Column names"""

    NODE_COLS = ["node_id", "easting", "northing"]
    EDGE_COLS = ["source", "target", "time_weighted", "wkt"]

    """NHS Constants"""

    NHS_URL = "https://files.digital.nhs.uk/assets/ods/current/"
    NHS_FILES = {
        "gp": "epraccur.zip",
        "dentists": "egdpprac.zip",
        "pharmacies": "epharmacyhq.zip",
    }


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def clean_postcodes(path: Path, current: bool) -> cudf.DataFrame:
    path = Config.RAW_DATA / "onspd/ONSPD_FEB_2021_UK.csv"
    postcodes = cudf.read_csv(
        path,
        usecols=["pcd", "oseast1m", "osnrth1m", "doterm", "ctry"],
        dtype={
            "pcd": "str",
            "oseast1m": "int",
            "osnrth1m": "int",
            "doterm": "str",
            "ctry": "str",
        },
    )
    postcodes = postcodes[
        postcodes["ctry"].isin(["E92000001", "W92000004", "S92000003"])
    ].drop("ctry", axis=1)
    postcodes = postcodes.rename(
        columns={"pcd": "postcode", "oseast1m": "easting", "osnrth1m": "northing"}
    )

    postcodes["postcode"] = postcodes["postcode"].str.replace(" ", "")
    postcodes["postcode"] = (
        postcodes.postcode.str[:-3] + " " + postcodes.postcode.str[-3:]
    )
    postcodes = postcodes.dropna(subset=["easting", "northing"])
    if current:
        return postcodes[postcodes["doterm"].isnull()].drop("doterm", axis=1)
    return postcodes.drop("doterm", axis=1)


def clean_retail_centres(path: Path) -> cudf.DataFrame:
    retail = gpd.read_file(path)
    retail["easting"], retail["northing"] = retail.geometry.x, retail.geometry.y
    retail = cudf.DataFrame(retail.drop("geometry", axis=1))
    retail = retail[["id", "easting", "northing"]]
    return retail


def clean_dentists(path: Path, postcodes: cudf.DataFrame) -> cudf.DataFrame:
    dentists = cudf.read_csv(path, usecols=[0, 9], header=None).drop_duplicates()
    dentists.rename(columns={"0": "dentist", "9": "postcode"}, inplace=True)
    dentists = dentists.merge(postcodes, on="postcode")  # type: ignore
    return dentists


def clean_gp(path: Path, postcodes: cudf.DataFrame, scot_path: Path) -> cudf.DataFrame:
    gp = cudf.read_csv(
        path,
        names=[
            "org_code",
            "name",
            "grouping",
            "geography",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "postcode",
            "open",
            "close",
            "status",
            "sub_type",
            "presc",
            "null",
        ],
    )
    gp = gp[gp["close"].isnull()][["org_code", "postcode"]].drop_duplicates()
    gp = gp.merge(postcodes, on="postcode")

    gp_scot = pd.ExcelFile(
        scot_path,
        engine="openpyxl",
    )
    gp_scot = pd.read_excel(gp_scot, "Practice Details", skiprows=5).dropna(how="all")
    gp_scot = gp_scot[["Practice Code", "Postcode"]].rename(
        columns={"Practice Code": "org_code", "Postcode": "postcode"}
    )
    gp_scot = cudf.from_pandas(gp_scot)
    gp_scot = gp_scot.merge(postcodes, on="postcode")
    gp_scot["org_code"] = gp_scot["org_code"].astype(str)
    gp = gp.append(gp_scot)
    return gp


def clean_pharmacies(path: Path, postcodes: cudf.DataFrame) -> cudf.DataFrame:
    pharmacies = cudf.read_csv(path, header=None, usecols=[0, 9])
    pharmacies.rename(columns={"0": "pharmacy", "9": "postcode"}, inplace=True)
    pharmacies = pharmacies.merge(postcodes, on="postcode")
    return pharmacies
