from typing import Optional, Union

import dask.dataframe as dd
from multiprocessing import Pool
from dask.distributed import Client, progress
import dask
import geopandas as gpd
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

from ahah.utils import Config, HiddenPrints


class RoutingCPU:
    def __init__(
        self,
        road_graph: pd.DataFrame,
        road_nodes: pd.DataFrame,
        postcodes: pd.DataFrame,
        pois: pd.DataFrame,
    ):
        self.postcode_ids: np.ndarray = postcodes["node_id"].drop_duplicates().values
        self.pois = pois.drop_duplicates(subset=["node_id"])

        self.road_graph = road_graph
        self.road_nodes = road_nodes

        self.dists = pd.DataFrame(
            {"node_id": self.postcode_ids, "distance": np.inf},
        )
        self.routes = []

    def fit(self) -> None:
        Parallel(n_jobs=-1)(
            delayed(self.get_shortest_dists)(poi) for _, poi in self.pois.iterrows()
        )

    def create_sub_graph(self, poi: pd.Series, buffer):
        breakpoint()
        poi_pt = poi[["easting", "northing"]]
        min_pt = np.array([poi_pt["easting"] - buffer, poi_pt["northing"] - buffer])
        max_pt = np.array([poi_pt["easting"] + buffer, poi_pt["northing"] + buffer])

        node_subset = self.road_nodes[
            np.all(
                (min_pt <= self.road_nodes[["easting", "northing"]].values)
                & (self.road_nodes[["easting", "northing"]].values <= max_pt),
                axis=1,
            )
        ]

        sub_edges = self.road_graph[
            self.road_graph["v"].isin(node_subset["node_id"])
            | self.road_graph["u"].isin(node_subset["node_id"])
        ]
        sub_graph = nx.Graph()
        sub_graph.add_weighted_edges_from(
            sub_edges[["u", "v", "length"]].to_records(index=False).tolist()
        )
        return sub_graph

    def get_shortest_dists(self, poi: pd.Series):
        sub_graph = self.create_sub_graph(poi=poi, buffer=poi["buffer"])
        shortest_paths = nx.single_source_dijkstra_path_length(
            sub_graph, source=poi["node_id"]
        )
        shortest_paths = pd.DataFrame(
            {"node_id": shortest_paths.keys(), "distance": shortest_paths.values()}
        )
        shortest_paths["node_id"] = shortest_paths["node_id"].astype("int32")

        pc_dist: pd.DataFrame = pd.DataFrame({"node_id": self.postcode_ids}).merge(
            shortest_paths[shortest_paths.node_id.isin(self.postcode_ids)].reset_index(
                drop=True
            ),
            how="outer",
        )  # type: ignore

        self.dists["distance"] = np.where(
            (pc_dist["distance"] < self.dists["distance"]),
            pc_dist["distance"],
            self.dists["distance"],
        )


if __name__ == "__main__":
    road_graph = (
        dd.read_csv(
            Config.OSM_GRAPH / "edges.csv",
            header=None,
            names=Config.EDGE_COLS,
            dtype={"u": "int32", "v": "int32", "length": "float32"},
        )
        .dropna(subset=["source", "target"])
        .fillna(value={"length": 0})
        .drop_duplicates()
        .compute()
    )

    road_nodes = pd.read_csv(
        Config.OSM_GRAPH / "nodes.csv",
        header=None,
        names=Config.NODE_COLS,
        dtype={"node_id": "int32", "easting": "float", "northing": "float"},
    ).drop_duplicates()

    postcodes = pd.read_csv(
        Config.PROCESSED_DATA / "postcodes.csv",
        dtype={
            "easting": "float",
            "northing": "float",
            "node_id": "int32",
        },
    )
    retail = pd.read_csv(
        Config.PROCESSED_DATA / "retail.csv",
        dtype={
            "easting": "float",
            "northing": "float",
            "node_id": "int32",
            "buffer": "float",
        },
    )
    dentists = pd.read_csv(
        Config.PROCESSED_DATA / "dentists.csv",
        dtype={
            "easting": "float",
            "northing": "float",
            "node_id": "int32",
            "buffer": "float",
        },
    )

    poi_dict = {"dentists": dentists, "retail": retail}
    for key, df in poi_dict.items():
        routing = RoutingCPU(
            road_graph=road_graph, road_nodes=road_nodes, postcodes=postcodes, pois=df
        )
        routing.fit()
        routing.dists.to_csv(Config.OUT_DATA / f"{key}_dist_nx.csv", index=False)
