from typing import List

import cudf
import cugraph
import cuspatial
import geopandas as gpd
import matplotlib.pyplot as plt
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
        pois: pd.DataFrame,
    ):
        self.postcode_ids: np.ndarray = postcodes["node_id"].unique().to_array()
        self.pois = pois.drop_duplicates("node_id")

        self.road_graph = road_graph
        self.road_nodes = road_nodes

        self.dists = pd.DataFrame()
        self.routes: List[gpd.GeoDataFrame] = []

    def fit(self, output_routes: bool = False) -> None:
        for poi in tqdm(self.pois.itertuples(), total=len(self.pois.index)):
            self.get_shortest_dists(poi, output_routes=output_routes)

        self.dists = self.dists.sort_values("distance").drop_duplicates("vertex")

    def create_sub_graph(self, poi) -> cugraph.Graph:
        buffer = poi.buffer
        while True:
            node_subset = cuspatial.points_in_spatial_window(
                min_x=poi.easting - buffer,
                max_x=poi.easting + buffer,
                min_y=poi.northing - buffer,
                max_y=poi.northing + buffer,
                xs=self.road_nodes["easting"],
                ys=self.road_nodes["northing"],
            )
            node_subset = node_subset.merge(
                self.road_nodes, left_on=["x", "y"], right_on=["easting", "northing"]
            ).drop(["x", "y"], axis=1)
            sub_edges = self.road_graph[
                self.road_graph["source"].isin(node_subset["node_id"])
                | self.road_graph["target"].isin(node_subset["node_id"])
            ]
            sub_graph = cugraph.Graph()
            sub_graph.from_cudf_edgelist(
                sub_edges,
                source="source",
                destination="target",
                edge_attr="time_weighted",
            )

            if (
                np.isin(poi.pc_ids, sub_graph.nodes().to_array()).sum()
                == poi.pc_ids.size
            ):
                return sub_graph
            buffer = (buffer + 100) * 2
            print(f"Increasing buffer size to {buffer}")

    def get_shortest_dists(self, poi, output_routes: bool = False):
        sub_graph = self.create_sub_graph(poi=poi)
        with HiddenPrints():
            shortest_paths: cudf.DataFrame = cugraph.filter_unreachable(
                cugraph.sssp(sub_graph, source=poi.node_id)
            )  # type: ignore

        pc_dist = shortest_paths[shortest_paths.vertex.isin(self.postcode_ids)]
        self.dists = self.dists.append(pc_dist.to_pandas())

        if output_routes:
            self.process_routes(pc_dist, shortest_paths, k=5)

    def process_routes(self, pc_dist, shortest_paths, k):
        dests = pc_dist.nsmallest(k, "distance")
        routes = []
        for _, dest in tqdm(dests.to_pandas().iterrows()):
            vert = int(dest["vertex"])
            route = []

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
        routes.crs = "epsg:27700"
        self.routes.append(routes)


if __name__ == "__main__":
    road_graph = cudf.read_parquet(Config.OSM_GRAPH / "edges.parquet")
    road_nodes = cudf.read_parquet(Config.OSM_GRAPH / "nodes.parquet")

    postcodes = cudf.read_parquet(Config.PROCESSED_DATA / "postcodes.parquet")
    retail = pd.read_parquet(Config.PROCESSED_DATA / "retail.parquet")
    dentists = pd.read_parquet(Config.PROCESSED_DATA / "dentists.parquet")
    gp = pd.read_parquet(Config.PROCESSED_DATA / "gp.parquet")

    poi_dict = {"retail": retail, "dentists": dentists, "gp": gp}

    for key, df in poi_dict.items():
        routing = Routing(
            road_graph=road_graph,
            road_nodes=road_nodes,
            postcodes=postcodes,
            pois=df,
        )
        routing.fit()

        dists = postcodes.merge(
            cudf.from_pandas(dists), left_on="node_id", right_on="vertex", how="left"
        ).drop(["vertex", "predecessor"], axis=1)
        dists.to_csv(Config.OUT_DATA / f"{key}_dist.csv", index=False)

        # temporary plots
        dists = dists[dists["distance"] != np.inf].dropna().to_pandas()
        plt.scatter(dists["easting"], dists["northing"], c=dists["distance"])
        plt.savefig(fname=f"{key}.png")
