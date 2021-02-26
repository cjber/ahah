import cudf
from cuml.neighbors.nearest_neighbors import NearestNeighbors

from ahah.utils import (
    Config,
    clean_dentists,
    clean_postcodes,
    clean_retail_centres,
    clean_gp,
)


def nearest_nodes(df: cudf.DataFrame, road_nodes: cudf.DataFrame) -> cudf.DataFrame:
    df = df.reset_index(drop=True)
    road_nodes = road_nodes.reset_index(drop=True)

    points = df[["easting", "northing"]]
    node_points = road_nodes[["easting", "northing"]]

    nbrs = NearestNeighbors(n_neighbors=1, output_type="cudf").fit(node_points)
    _, indices = nbrs.kneighbors(points)

    nodes = road_nodes.iloc[indices].reset_index(drop=True)
    return nodes.join(df.drop(["easting", "northing"], axis=1))


def get_buffers(
    poi: cudf.DataFrame, postcodes: cudf.DataFrame, k: int
) -> cudf.DataFrame:
    poi = poi.reset_index(drop=True)
    postcodes = postcodes.reset_index(drop=True)
    poi_pts = poi[["easting", "northing"]]
    pc_pts = postcodes[["easting", "northing"]]

    nbrs = NearestNeighbors(n_neighbors=k, output_type="cudf").fit(poi_pts)
    distances, indices = nbrs.kneighbors(pc_pts)

    poi = (
        postcodes.join(indices)[["node_id"] + indices.columns.tolist()]
        .set_index("node_id")
        .stack()
        .rename("node_id")
        .reset_index()
        .rename(columns={0: "pc_ids"})
        .drop(1, axis=1)
        .groupby("node_id")
        .agg(list)
        .join(poi, how="right")
    )

    distances = distances.stack().rename("dist").reset_index().drop(1, axis=1)
    indices = indices.stack().rename("ind").reset_index().drop(1, axis=1)

    poi_nodes = (
        poi[["node_id"]].iloc[indices["ind"].values]["node_id"].reset_index(drop=True)
    )
    buffers = cudf.DataFrame({"node_id": poi_nodes, "buffer": distances["dist"].values})
    # TODO: groupby may be better?
    buffers = buffers.sort_values("buffer", ascending=False).drop_duplicates("node_id")

    # this will drops rows that did not appear in the KNN
    # may change this
    poi = poi.merge(buffers, on="node_id", how="left").dropna()
    return poi


if __name__ == "__main__":
    road_nodes = cudf.read_parquet(Config.OSM_GRAPH / "nodes.parquet")
    postcodes: cudf.DataFrame = clean_postcodes(path=Config.RAW_DATA / "postcodes")

    retail: cudf.DataFrame = clean_retail_centres(
        path=Config.RAW_DATA / "retailcentrecentroids.gpkg"
    )
    dentists: cudf.DataFrame = clean_dentists(
        path=Config.RAW_DATA / "nhs-dent-stat-eng-19-20-anx3-act.csv",
        postcodes=postcodes,
    )
    gp = clean_gp(
        path=Config.RAW_DATA / "epraccur.csv",
        postcodes=postcodes,
        scot_path=Config.RAW_DATA / "Practice_ContactDetails_Jan2021_v2.xlsx",
    )

    postcodes = nearest_nodes(df=postcodes, road_nodes=road_nodes)
    retail = nearest_nodes(df=retail, road_nodes=road_nodes)
    dentists = nearest_nodes(df=dentists, road_nodes=road_nodes)
    gp = nearest_nodes(df=gp, road_nodes=road_nodes)

    retail = get_buffers(poi=retail, postcodes=postcodes, k=10)
    dentists = get_buffers(poi=dentists, postcodes=postcodes, k=10)
    gp = get_buffers(poi=gp, postcodes=postcodes, k=10)

    postcodes.to_parquet(Config.PROCESSED_DATA / "postcodes.parquet", index=False)
    retail.to_parquet(Config.PROCESSED_DATA / "retail.parquet", index=False)
    dentists.to_parquet(Config.PROCESSED_DATA / "dentists.parquet", index=False)
    gp.to_parquet(Config.PROCESSED_DATA / "gp.parquet", index=False)
