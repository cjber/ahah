import time

import cudf
from cuml.neighbors import NearestNeighbors

from ahah.utils import Config, clean_postcodes


postcodes: cudf.DataFrame = clean_postcodes(path=Config.RAW_DATA / "postcodes")
road_nodes = cudf.read_csv(
    Config.OSM_GRAPH / "nodes.csv",
    header=None,
    names=Config.NODE_COLS
)

pc_points = postcodes[['easting', 'northing']].to_pandas()
node_points = road_nodes[['easting', 'northing']].to_pandas()

nbrs = NearestNeighbors(n_neighbors=1).fit(node_points)

t1 = time.perf_counter()
distances, indices = nbrs.kneighbors(pc_points)
t2 = time.perf_counter()
time_taken = t2-t1
print(time_taken / 60)

postcodes['node_id'] = road_nodes.iloc[indices.flatten()]['id'].values

postcodes.to_csv(Config.PROCESSED_DATA / 'postcodes.csv', index=False)
