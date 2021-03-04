import geopandas as gpd
import fiona

fiona.listlayers(
    "data/raw/MasterMap Highways Network_3968482/Highways_Data_March19.gdb.zip"
)

nodes = gpd.read_file(
    "data/raw/MasterMap Highways Network_3968482/Highways_Data_March19.gdb.zip",
    driver="FileGDB",
    layer="RoadNode",
)

fiona.listlayers(
    "data/raw/MasterMap Highways Network_3968482/Highways_Network_March19.gdb.zip"
)

edges = gpd.read_file(
    "data/raw/MasterMap Highways Network_3968482/Highways_Network_March19.gdb.zip",
    driver="FileGDB",
    layer="RoadLink_N",
)

import matplotlib.pyplot as plt

gdf.plot()
plt.show()
