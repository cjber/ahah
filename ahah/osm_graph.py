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
    # converts to time in minutes rather than distance
    speed_dict = {6: 70, 5: 70, 4: 60, 3: 60, 2: 60, 1: 30, 0: 20}
    speed_dict = {key: value * 1.609344 for key, value in speed_dict.items()}
    edges["speed_estimate"] = edges["car_forward"].map(speed_dict)
    edges["time_weighted"] = ((edges["length"] / 1000) / edges["speed_estimate"]) * 60

    nodes.rename(columns={"id": "node_id"}, inplace=True)

    # change high int values to lower sequential ones as high int breaks with int32
    id_mapping = (
        nodes["node_id"]
        .reset_index()
        .set_index("node_id")["index"]
        .to_pandas()
        .to_dict()
    )
    nodes["node_id"] = nodes["node_id"].map(id_mapping)
    edges["source"] = edges["source"].map(id_mapping)
    edges["target"] = edges["target"].map(id_mapping)

    nodes["node_id"] = nodes["node_id"].astype("int32")
    edges = edges.astype({"source": "int32", "target": "int32"})

    # convert to BNG
    transformer: Transformer = Transformer.from_crs(4326, 27700)
    lon = nodes["lon"].values
    lat = nodes["lat"].values
    bng = [
        transformer.transform(y, x) for (x, y) in tqdm(zip(lon, lat), total=len(nodes))
    ]
    nodes = nodes.join(cudf.DataFrame(bng, columns=["easting", "northing"]))
    return nodes, edges


if __name__ == "__main__":
    gb_pbf = Path(Config.RAW_DATA / "osm_pbf" / "great-britain-latest.osm.pbf")
    if not gb_pbf.is_file():
        get_gb_graph(gb_pbf)

    nodes = cudf.read_csv(Config.RAW_DATA / "osm_pbf/nodes.csv")
    edges = cudf.read_csv(Config.RAW_DATA / "osm_pbf/edges.csv")

    nodes, edges = process_graph(nodes, edges)

    nodes[Config.NODE_COLS].to_parquet(
        Config.OSM_GRAPH / "nodes.parquet",
        index=False,
        dtypes=["int32", "float", "float"],
    )
    edges[Config.EDGE_COLS].to_parquet(
        Config.OSM_GRAPH / "edges.parquet",
        index=False,
        dtypes=["int32", "int", "int", "str"],
    )
