from pyrosm import OSM, get_data
from pyrosm.data import sources
from tqdm import tqdm

from ahah.utils import Config


def process_pbf(pbf_name: str, network_type: str = "walking") -> None:
    """
    Read OSM pbf and create network graph, saves to csvs.

    :param pbf_name str: Name of pbf (English counties + wales and scotland)
    :param network_type str: walking/driving etc.
    """
    osm = OSM(get_data(pbf_name, directory=Config.RAW_DATA / "osm_pbf"))
    nodes, edges = osm.get_network(  # type: ignore
        network_type=network_type, nodes=True
    )

    # convert from mercator to BNG for use with euclidean dists
    crs = "EPSG:4326"
    crs_bng = "EPSG:27700"
    nodes.crs = crs
    edges.crs = crs
    nodes = nodes.to_crs(crs_bng)
    edges = edges.to_crs(crs_bng)

    edges[Config.EDGE_COLS].to_csv(
        Config.OSM_GRAPH / "edges.csv", mode="a", header=False, index=False
    )
    nodes = nodes.rename(columns={"id": "node_id"})
    nodes["easting"], nodes["northing"] = nodes.geometry.x, nodes.geometry.y
    nodes[Config.NODE_COLS].to_csv(
        Config.OSM_GRAPH / "nodes.csv", mode="a", header=False, index=False
    )


if __name__ == "__main__":
    # contains english counties + england, scotland and wales
    counties = sources.subregions.great_britain.available
    counties = [
        county for county in counties if county not in ["england", "scotland"]
    ]

    for county in tqdm(counties):
        process_pbf(pbf_name=county)
