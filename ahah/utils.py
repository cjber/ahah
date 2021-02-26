import os
from pathlib import Path
import sys

import cudf
import dask_cudf
import geopandas as gpd


class Config:
    DATA_PATH = Path("data/")
    RAW_DATA = DATA_PATH / "raw"
    PROCESSED_DATA = DATA_PATH / "processed"
    OUT_DATA = DATA_PATH / "out"
    OSM_GRAPH = PROCESSED_DATA / "osm"

    NODE_COLS = ["node_id", "easting", "northing"]
    EDGE_COLS = ["source", "target", "time_weighted"]


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def clean_postcodes(path: Path, exclude_scotland: bool = False) -> cudf.DataFrame:
    pc_path = Path(path).glob("**/*.csv")
    postcodes = dask_cudf.read_csv(list(pc_path), header=None, usecols=[0, 2, 3])
    postcodes = postcodes.rename(  # type: ignore
        columns={"0": "postcode", "2": "easting", "3": "northing"}
    )
    postcodes = postcodes[(postcodes.easting != 0) & (postcodes.northing != 0)]
    postcodes = postcodes.drop_duplicates(subset=["easting", "northing"])
    postcodes = postcodes.compute()

    if exclude_scotland:
        postcodes = postcodes[
            ~postcodes["postcode"]
            .str[:2]
            .isin(
                [
                    "AB",
                    "DD",
                    "DG",
                    "EH",
                    "FK",
                    "HS",
                    "IV",
                    "KA",
                    "KW",
                    "KY",
                    "ML",
                    "PA",
                    "PH",
                    "TD",
                    "ZE",
                ]
            )
        ]
        postcodes = postcodes[~postcodes["postcode"].str.contains("^G[0-9].*")]
    return postcodes


def clean_retail_centres(path: Path) -> cudf.DataFrame:
    retail = gpd.read_file(path)
    retail["easting"], retail["northing"] = retail.geometry.x, retail.geometry.y
    retail = cudf.DataFrame(retail.drop("geometry", axis=1))
    retail = retail[["id", "easting", "northing"]]
    return retail


def clean_dentists(path: Path, postcodes: cudf.DataFrame) -> cudf.DataFrame:
    dentists = cudf.read_csv(path)
    dentists = dentists[["PRACTICE_CODE", "PRAC_POSTCODE"]].drop_duplicates()
    dentists.rename(columns={"PRAC_POSTCODE": "postcode"}, inplace=True)
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
    gp = gp[gp["close"].isnull()]
    gp = gp[["org_code", "postcode"]].drop_duplicates()
    gp = gp.merge(postcodes, on="postcode")

    import pandas as pd

    gp_scot = pd.ExcelFile(
        Config.RAW_DATA / "Practice_ContactDetails_Jan2021_v2.xlsx",
        engine="openpyxl",
    )
    gp_scot = pd.read_excel(gp_scot, "Practice Details", skiprows=5)
    gp_scot = gp_scot[["Practice Code", "Postcode"]].rename(
        columns={"Practice Code": "org_code", "Postcode": "postcode"}
    )
    gp_scot = cudf.from_pandas(gp_scot)
    gp_scot = gp_scot.merge(postcodes, on="postcode")
    gp_scot["org_code"] = gp_scot["org_code"].astype(str)

    gp = gp.append(gp_scot)
    return gp
