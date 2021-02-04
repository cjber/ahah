import cudf
import cugraph
from cuml.neighbors import NearestNeighbors
import dask_cudf
import geopandas as gpd
import numpy as np
import pandas as pd
from rich.progress import track
from shapely.geometry import LineString

from ahah.utils import Config, clean_retail_centres


class Routing:
    def __init__(
            self,
            road_graph: cudf.DataFrame,
            road_nodes: cudf.DataFrame,
            postcode_ids: np.ndarray,
            buffer=None
    ):
        self.road_nodes = road_nodes
        self.road_graph = road_graph

        self.postcodes = self.road_nodes[
            self.road_nodes['id'].isin(postcode_ids)
        ]
        assert len(self.postcodes.index) == len(postcode_ids)

        self.create_knn_objects()
        self.create_graph()

    def create_knn_objects(self):
        self.node_nbrs = NearestNeighbors(n_neighbors=1).fit(
            self.road_nodes[['easting', 'northing']]
        )
        self.pc_nbrs = NearestNeighbors(n_neighbors=10).fit(
            self.postcodes[['easting', 'northing']]
        )

    def create_graph(self):
        self.graph = cugraph.Graph()
        self.graph.from_cudf_edgelist(
            self.road_graph,
            source='u',
            destination='v',
            edge_attr='length'
        )

    def fit(self, poi):
        breakpoint()
        _, poi_node = self.node_nbrs.kneighbors(
            np.array(poi).reshape(1, -1)
        )
        _, pcs = self.pc_nbrs.kneighbors(
            np.array(poi).reshape(1, -1)
        )

        pc_bfs = cugraph.bfs(
            self.graph,
            start=int(poi_node['id'].values),
            return_predecessors=True
        )
        pc_bfs = pc_bfs[pc_bfs['vertex'].isin(pcs)]
        #  predecessor = int(poi_node['predecessor'].values)
        #   route = []
        #    while predecessor != -1:
        #         route.append(predecessor)
        #         prev_node = pc_bfs[pc_bfs['vertex'] == predecessor]
        #         predecessor = int(prev_node['predecessor'].values)
        #     routes.append(route)
        #     poi_node_dists.append(pc_bfs)
        # return poi_node_dists, routes


road_graph = (
    dask_cudf.read_csv(
        Config.OSM_GRAPH / 'edges.csv',
        header=None,
        names=Config.EDGE_COLS,
        dtype={'u': 'int32', 'v': 'int32', 'length': 'float32'}
    )
    .dropna(subset=['u', 'v'])  # type:ignore
    .fillna(value={'length': 0})
    .compute()
)
road_graph = cudf.read_csv(Config.OSM_GRAPH / 'edges.csv')
road_nodes = cudf.read_csv(
    Config.OSM_GRAPH / 'nodes.csv',
    header=None,
    names=Config.NODE_COLS
)

postcodes = cudf.read_csv(Config.PROCESSED_DATA / 'postcodes.csv')
postcode_ids = postcodes['node_id'].to_pandas().values
retail = clean_retail_centres(Config.RAW_DATA / 'retailcentrecentroids.gpkg')
retail_point: pd.Series = retail.iloc[1]  # type:ignore

pc_dist = Routing(
    road_nodes=road_nodes,
    road_graph=road_graph,
    postcode_ids=postcode_ids
)
pc_dist.fit(poi=retail_point)


# for idx, pc in track(postcodes[100_000:].to_pandas().iterrows(), total=len(postcodes)):
#     pc_dist = PostCodeDistances(
#         postcode=pc,
#         road_graph=road_graph,
#         road_nodes=road_nodes,
#         poi=hospitals,
#         #        buffer=10_000
#     )
#     road_filtered = pc_dist.road_nodes
#     poi_node_dists, routes = pc_dist.fit()
#     break

# fig, ax = plt.subplots()
# roads_gpd = gpd.GeoDataFrame(road_filtered.to_pandas(), geometry=gpd.points_from_xy(
#     road_filtered['easting'].to_pandas(), road_filtered['northing'].to_pandas()
# ))
# hp_gpd = gpd.GeoDataFrame(hospitals.to_pandas(), geometry=gpd.points_from_xy(
#     hospitals['easting'].to_pandas(), hospitals['northing'].to_pandas()))
# routes_gpd = road_nodes[road_nodes['nodeID'].isin(routes[-1])].to_pandas()
# route_geom = [xy for xy in zip(routes_gpd.easting, routes_gpd.northing)]
# route_line = LineString(route_geom)
# route_gpd = gpd.GeoSeries()
# route_gpd['geometry'] = route_line
# roads_gpd.plot(ax=ax)
# hp_gpd.plot(ax=ax, color='orange')
# route_gpd.plot(ax=ax, color='red')
# plt.xlim([min(roads_gpd['easting']), max(roads_gpd['easting'])])
# plt.ylim([min(roads_gpd['northing']), max(roads_gpd['northing'])])
# plt.show()

# def haversine_single_point(
#         self,
#         multiple: cudf.DataFrame,
#         point: pd.Series,
#         k: int
# ) -> Union[cudf.DataFrame, None]:
#     multiple['easting_REF'] = int(point['easting'])
#     multiple['northing_REF'] = int(point['northing'])
#     multiple['distance'] = cuspatial.haversine_distance(
#         multiple['easting'],
#         multiple['northing'],
#         multiple['easting_REF'],
#         multiple['northing_REF']
#     )
#     multiple.drop(
#         ['easting_REF', 'northing_REF'], axis=1, inplace=True
#     )
#     nearest = multiple.nsmallest(k, 'distance')
#     multiple.drop(['distance'], axis=1, inplace=True)
#     return nearest
