from rich.progress import track

from ahah.utils import Config
import geopandas as gpd

if __name__ == '__main__':
    osm_nodes = (Config.RAW_DATA / 'osm_gpkg').glob('*_node.gpkg')
    osm_edges = (Config.RAW_DATA / 'osm_gpkg').glob('*_edge.gpkg')

    nodes_csv = Config.OSM_GRAPH / 'nodes.csv'
    edges_csv = Config.OSM_GRAPH / 'edges.csv'

    with nodes_csv.open(mode='a') as csv:
        for gpkg in track(
                osm_nodes,
                description='Nodes...',
                total=49
        ):
            gpd.read_file(gpkg)[Config.NODE_COLS].to_csv(
                csv, header=False, index=False  # type: ignore
            )

    with edges_csv.open('a') as csv:
        for gpkg in track(
                osm_edges,
                description='Edges...',
                total=49
        ):
            gpd.read_file(gpkg)[Config.EDGE_COLS].to_csv(
                csv, header=False, index=False  # type: ignore
            )
