import cudf
from ahah.utils import Config

pc_to_lsoa = cudf.read_csv(Config.OUT_DATA / "PCD_OA_LSOA_MSOA_LAD_AUG19_UK_LU.zip")

dentists = cudf.read_csv(Config.OUT_DATA / "gp_dist.csv")
dentists = dentists.merge(pc_to_lsoa, left_on="postcode", right_on="pcd7")
dentists = dentists.groupby("lsoa11cd")["distance"].mean().reset_index()

ahahv2 = cudf.read_csv(Config.OUT_DATA / "allvariableslsoawdeciles.csv")
temp = ahahv2[["lsoa11", "gpp_dist"]].merge(
    dentists[["lsoa11cd", "distance"]], left_on="lsoa11", right_on="lsoa11cd"
)

temp["dist_diff"] = temp["distance"] - temp["gpp_dist"]

# temp = temp[~temp["lsoa11"].str.startswith("S")]
# temp = temp[~temp["lsoa11"].str.startswith("W")]
temp.sort_values(ascending=False, by="dist_diff")
