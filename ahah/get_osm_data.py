from pyrosm import OSM, get_data
from pyrosm.data import sources
from rich.progress import track

from ahah.utils import Config


def get_county(county: str, network_type='driving') -> None:
    print(f'Processing {county}.')
    osm = OSM(get_data(county, directory=Config.RAW_DATA / 'osm_pbf'))
    nodes, edges = osm.get_network(  # type: ignore
        network_type=network_type,
        nodes=True
    )

    crs = 'EPSG:4326'
    crs_bng = 'EPSG:27700'
    for gdf in (nodes, edges):
        gdf.crs = crs
        gdf = edges.to_crs(crs_bng)

    edges[Config.EDGE_COLS].to_csv(
        Config.OSM_GRAPH / 'edges.csv',
        mode='a',
        header=False,
        index=False
    )
    nodes['easting'], nodes['northing'] = nodes.geometry.x, nodes.geometry.y
    nodes[Config.NODE_COLS].to_csv(
        Config.OSM_GRAPH / 'nodes.csv',
        mode='a',
        header=False,
        index=False
    )


if __name__ == '__main__':
    # contains english counties + england, scotland and wales
    counties = sources.subregions.great_britain.available
    counties.remove('england')  # included as counties

    for county in track(counties):
        get_county(county=county)
