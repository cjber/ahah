from pathlib import Path
from subprocess import Popen

import cudf
from pyproj import Transformer
from pyrosm import OSM, get_data
from tqdm import tqdm

from ahah.utils import Config


def get_gb_graph(gb_pbf):
    OSM(get_data("great_britain", directory=gb_pbf.parents[0]))
    p = Popen(["osm4routing", gb_pbf.name], cwd=gb_pbf.parents[0])
    p.wait()


def process_graph(nodes, edges):
    # for now exclude one way streets for ease
    edges = edges[(edges["car_forward"] != 0) | (edges["car_backward"] != 0)]
    # remove nodes that aren't in the filtered graph
    nodes = nodes[nodes["id"].isin(edges["source"]) | nodes["id"].isin(edges["target"])]

    # converts to time in minutes rather than distance
    speed_dict = {6: 70, 5: 70, 4: 60, 3: 60, 2: 60, 1: 30}
    speed_dict = {key: value * 1.609344 for key, value in speed_dict.items()}
    edges["speed_estimate"] = edges["car_forward"].map(speed_dict)
    edges["time_weighted"] = ((edges["length"] / 1000) / edges["speed_estimate"]) * 60

    # convert to BNG
    lon = nodes["lon"].values
    lat = nodes["lat"].values
    transformer: Transformer = Transformer.from_crs(4326, 27700)
    bng = [
        transformer.transform(y, x)
        for i, (x, y) in tqdm(enumerate(zip(lon, lat)), total=len(lat), ascii=True)
    ]

    nodes = nodes.reset_index(drop=True).join(
        cudf.DataFrame(bng, columns=["easting", "northing"])
    )
    nodes.rename(columns={"id": "node_id"}, inplace=True)
    return nodes, edges


if __name__ == "__main__":
    gb_pbf = Path(Config.RAW_DATA / "osm_pbf" / "great-britain-latest.osm.pbf")
    if not gb_pbf.is_file():
        get_gb_graph(gb_pbf)

    nodes = cudf.read_csv(Config.RAW_DATA / "osm_pbf/nodes.csv")
    edges = cudf.read_csv(Config.RAW_DATA / "osm_pbf/edges.csv")

    nodes, edges = process_graph(nodes, edges)

    nodes[Config.NODE_COLS].to_csv(Config.OSM_GRAPH / "nodes.csv", index=False)
    edges[Config.EDGE_COLS].to_csv(Config.OSM_GRAPH / "edges.csv", index=False)
