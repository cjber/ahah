import cudf
import matplotlib.pyplot as plt
import pandas as pd

from ahah.routing import Routing
from ahah.utils import Config

road_graph = cudf.read_parquet(Config.OSM_GRAPH / "edges.parquet")
road_nodes = cudf.read_parquet(Config.OSM_GRAPH / "nodes.parquet")

postcodes = cudf.read_parquet(Config.PROCESSED_DATA / "postcodes.parquet")
gp = pd.read_parquet(Config.PROCESSED_DATA / "gp.parquet")


routing = Routing(
    road_graph=road_graph,
    road_nodes=road_nodes,
    postcodes=postcodes,
    pois=gp.iloc[:1],
)
routing.fit(output_routes=True)


test = routing.routes[0]
test.plot()
plt.show()
