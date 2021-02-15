# GPU accelerated routing with RAPIDS

This repository contains the python code used to find distances to various health related points of interest from each postcode within Great Britain.

## Workflow

### Get Open Street Map data `ahah/get_osm_data.py`

_NOTE: This may be overhauled to use the `great_britain` data dump. For now this uses too much memory._

Open Street Map produces `pbf` files containing information regarding the road network. This step uses the `pyrosm` python library to download and process these `pbf` files for each county within England, Scotland, and Wales.

* Download `pbf` files
* Get `nodes` and `edges` for road network
* Convert `crs` from `EPSG:4326` to `EPSG:27700` British National Grid, which uses a planar coordinate system
* Write `nodes` and `edges` for each `pbf` into combined files

### Process Data `ahah/process_data.py`

This stage prepares the `nodes`, `postcodes`, and `poi` data for use in RAPIDS `cugraph`. This stage also makes use of utility functions to assist with data preparation from the raw data sources.

* Clean raw data
* Find nearest road node to each postcode and point of interest using GPU accelerated K Means Clustering
* Determine minimum buffer distance to use for each point of interest
  * Nearest 10 points of interest to each postcode using K Means
  * Find distances returned for each POI neighbour
  * For each POI the maximum distance to associated postcodes is taken and saved as a buffer for this POI
* All processed data written to respective files

### Routing `ahah/routing.py`

The routing stage of this project primarily makes use of the RAPIDS `cugraph` library. This stage iterates sequentially over each POI of a certain type and finds routes to every postcode within a certain buffer.

* Iterate over POI of a certain type
* Using POI buffer value subset road nodes using buffer value centred on POI
  * Uses `cuspatial.points_in_spatial_window`
* Subset converted into `cugraph.Graph()`
* Run single source shortest path from POI to each node in the sub graph
  * `sssp` takes into account `weights`, which in this case are the `length` of each connection between nodes as reported by OSM.
* `SSSP` distances subset by postcode nodes, these distances are added to a complete dataframe of postcodes if they are smaller than existing values

