from typing import Optional, Union

import cudf
import cugraph
import cuspatial
import dask_cudf
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

from ahah.utils import Config, HiddenPrints


class Routing:
    def __init__(
        self,
        road_graph: cudf.DataFrame,
        road_nodes: cudf.DataFrame,
        postcodes: cudf.DataFrame,
        pois: cudf.DataFrame,
    ):
        self.postcode_ids: np.ndarray = (
            postcodes["node_id"].drop_duplicates().to_pandas().values
        )
        self.pois: pd.DataFrame = pois.drop_duplicates(subset=["node_id"]).to_pandas()  # type: ignore

        self.road_graph = road_graph
        self.road_nodes = road_nodes

        self.dists = cudf.DataFrame(
            {"node_id": self.postcode_ids, "distance": np.inf},
            dtype={"node_id": "int32", "distance": "float"},
        )
        self.routes = []

    def fit(self, output_routes: bool = False) -> None:
        for _, poi in tqdm(self.pois.iterrows(), total=len(self.pois)):
            self.get_shortest_dists(poi, output_routes=False)

    def create_sub_graph(self, poi: pd.Series, buffer) -> cugraph.Graph:
        poi_pt: pd.Series = poi[["easting", "northing"]]

        node_subset = cuspatial.points_in_spatial_window(
            min_x=poi_pt["easting"] - buffer,
            max_x=poi_pt["easting"] + buffer,
            min_y=poi_pt["northing"] - buffer,
            max_y=poi_pt["northing"] + buffer,
            xs=self.road_nodes["easting"],
            ys=self.road_nodes["northing"],
        )
        node_subset = node_subset.merge(
            self.road_nodes, left_on=["x", "y"], right_on=["easting", "northing"]
        ).drop(["x", "y"], axis=1)

        sub_edges = self.road_graph[
            self.road_graph["v"].isin(node_subset["node_id"])
            | self.road_graph["u"].isin(node_subset["node_id"])
        ]
        sub_graph = cugraph.Graph()
        sub_graph.from_cudf_edgelist(
            sub_edges,
            source="u",
            destination="v",
            edge_attr="length",
        )
        return sub_graph

    def get_shortest_dists(self, poi: pd.Series, output_routes: bool = False):
        sub_graph = self.create_sub_graph(poi=poi, buffer=poi["buffer"])
        with HiddenPrints():
            shortest_paths: cudf.DataFrame = cugraph.filter_unreachable(
                cugraph.sssp(sub_graph, source=poi["node_id"])
            )  # type: ignore

        pc_dist: cudf.DataFrame = cudf.DataFrame({"vertex": self.postcode_ids}).merge(
            shortest_paths[shortest_paths.vertex.isin(self.postcode_ids)].reset_index(
                drop=True
            ),
            how="outer",
        )  # type: ignore

        self.dists["distance"] = np.where(
            (pc_dist["distance"].to_pandas() < self.dists["distance"].to_pandas()),
            pc_dist["distance"].to_pandas(),
            self.dists["distance"].to_pandas(),
        )

        if output_routes:
            self.process_routes(pc_dist, shortest_paths, k=5)

    def process_routes(self, pc_dist, shortest_paths, k):
        dests = pc_dist.drop_duplicates().nsmallest(k, "distance")
        routes = []
        for _, dest in tqdm(dests.to_pandas().iterrows()):
            vert = int(dest["vertex"])
            route = []

            # TODO: order seems random but nodes are correct
            while vert != -1:
                vert = int(
                    shortest_paths[shortest_paths["vertex"] == vert][
                        "predecessor"
                    ].values
                )
                if vert != -1:
                    route.append(vert)
                else:
                    route.append(int(dest["vertex"]))

            if len(route) > 1:
                route_df: cudf.DataFrame = (
                    cudf.DataFrame({"node_id": route})
                    .merge(self.road_nodes)
                    .to_pandas()  # type:ignore
                )
                route_geom = [xy for xy in zip(route_df.easting, route_df.northing)]
                route_gpd = gpd.GeoSeries()
                route_gpd["geometry"] = LineString(route_geom)
                routes.append(route_gpd)
        routes = gpd.GeoDataFrame(routes)
        self.routes.append(routes)


if __name__ == "__main__":
    road_graph = (
        dask_cudf.read_csv(
            Config.OSM_GRAPH / "edges.csv",
            header=None,
            names=Config.EDGE_COLS,
            dtype={"u": "int32", "v": "int32", "length": "float32"},
        )
        .dropna(subset=["u", "v"])  # type:ignore
        .fillna(value={"length": 0})
        .drop_duplicates()
        .compute()
    )
    road_nodes = cudf.read_csv(
        Config.OSM_GRAPH / "nodes.csv",
        header=None,
        names=Config.NODE_COLS,
        dtype={"node_id": "int32", "easting": "int64", "northing": "int64"},
    ).drop_duplicates()

    postcodes = cudf.read_csv(
        Config.PROCESSED_DATA / "postcodes.csv",
        dtype={
            "easting": "int64",
            "northing": "int64",
            "node_id": "int32",
        },
    )
    retail = cudf.read_csv(
        Config.PROCESSED_DATA / "retail.csv",
        dtype={
            "easting": "int64",
            "northing": "int64",
            "node_id": "int32",
            "buffer": "int64",
        },
    )
    dentists = cudf.read_csv(
        Config.PROCESSED_DATA / "dentists.csv",
        dtype={
            "easting": "int64",
            "northing": "int64",
            "node_id": "int32",
            "buffer": "int64",
        },
    )

    poi_dict = {"dentists": dentists, "retail": retail}
    for key, df in poi_dict.items():
        routing = Routing(
            road_graph=road_graph, road_nodes=road_nodes, postcodes=postcodes, pois=df
        )
        routing.fit()
        routing.dists.to_csv(Config.OUT_DATA / f"{key}_dist.csv", index=False)
