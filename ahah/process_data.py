import cudf
from cuml.neighbors import NearestNeighbors

from ahah.utils import Config, clean_dentists, clean_postcodes, clean_retail_centres


def nearest_nodes(
    df: cudf.DataFrame,
    road_nodes: cudf.DataFrame,
) -> cudf.DataFrame:
    """
    Get nearest road node to each point in a poi df using K Means

    :param df cudf.DataFrame: Input df with easting/northing
    :param road_nodes cudf.DataFrame: Road nodes list with easting/northing
    """
    points = df[["easting", "northing"]]
    node_points = road_nodes[["easting", "northing"]]

    nbrs = NearestNeighbors(n_neighbors=1, output_type="numpy").fit(node_points)
    distances, indices = nbrs.kneighbors(points)

    return road_nodes.iloc[indices.flatten()].reset_index(drop=True)


def get_buffers(
    poi: cudf.DataFrame, postcodes: cudf.DataFrame, k: int
) -> cudf.DataFrame:
    poi_pts = poi[["easting", "northing"]]
    pc_pts = postcodes[["easting", "northing"]]

    nbrs = NearestNeighbors(n_neighbors=k, output_type="numpy").fit(poi_pts)
    distances, indices = nbrs.kneighbors(pc_pts)

    distances = distances.flatten()
    indices = indices.flatten()

    poi_nodes = poi.iloc[indices]["node_id"].reset_index(drop=True)
    buffer_cdf = cudf.DataFrame({"node_id": poi_nodes, "buffer": distances})

    buffer_cdf = buffer_cdf.sort_values("buffer", ascending=False).drop_duplicates(
        "node_id"
    )
    # assert buffer_cdf.buffer.min() != 0  # don't have 0 distance buffers
    return poi.merge(buffer_cdf, on="node_id")  # type: ignore


if __name__ == "__main__":
    road_nodes = cudf.read_csv(Config.OSM_GRAPH / "nodes.csv")

    postcodes: cudf.DataFrame = clean_postcodes(
        path=Config.RAW_DATA / "postcodes", exclude_scotland=False
    )
    retail: cudf.DataFrame = clean_retail_centres(
        path=Config.RAW_DATA / "retailcentrecentroids.gpkg"
    )
    dentists: cudf.DataFrame = clean_dentists(
        path=Config.RAW_DATA / "nhs-dent-stat-eng-19-20-anx3-act.csv",
        postcodes=postcodes,
    )

    postcodes = nearest_nodes(
        df=postcodes,
        road_nodes=road_nodes,
    )
    dentists = nearest_nodes(
        df=dentists,
        road_nodes=road_nodes,
    )
    retail = nearest_nodes(
        df=retail,
        road_nodes=road_nodes,
    )
    retail = get_buffers(poi=retail, postcodes=postcodes, k=1)
    dentists = get_buffers(poi=dentists, postcodes=postcodes, k=1)

    dentists.to_csv(Config.PROCESSED_DATA / "dentists.csv", index=False)
    retail.to_csv(Config.PROCESSED_DATA / "retail.csv", index=False)
    postcodes.to_csv(Config.PROCESSED_DATA / "postcodes.csv", index=False)
