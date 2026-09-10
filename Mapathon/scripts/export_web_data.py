"""Export Chennai Mapathon QGIS layers to WGS84 GeoJSON and data.js for web hosting."""

from pathlib import Path
import json
import geopandas as gpd

GPKG_PATH = Path("qgis/chennai_mapathon_layers_desktop.gpkg")
WEB_DIR = Path("web")
DATA_DIR = WEB_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = {
    "corridors": "chennai_corridor_guides",
    "stations": "chennai_stations_predictions",
    "catchments": "chennai_catchments_800m",
    "lulc": "bhuvan_lulc_1516",
}

metro_data = {}

print("Reading and reprojecting layers to EPSG:4326 (WGS84)...")

for key, layer_name in LAYERS.items():
    print(f"  Processing {key} ({layer_name})...")
    gdf = gpd.read_file(GPKG_PATH, layer=layer_name)
    gdf_4326 = gdf.to_crs(epsg=4326)
    
    # Export individual GeoJSON file
    geojson_str = gdf_4326.to_json()
    geojson_obj = json.loads(geojson_str)
    metro_data[key] = geojson_obj
    
    out_file = DATA_DIR / f"{key}.geojson"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f)
    
    file_size_kb = out_file.stat().st_size / 1024
    print(f"    -> Saved {out_file} ({len(gdf)} features, {file_size_kb:.1f} KB)")

# Write standalone data.js bundle for CORS-free offline / direct hosting
data_js_path = WEB_DIR / "data.js"
print(f"Writing standalone JS bundle: {data_js_path}...")
with open(data_js_path, "w", encoding="utf-8") as f:
    f.write("// Chennai Metro Phase 1 & 2 + ISRO Bhuvan Spatial Dataset\n")
    f.write("window.METRO_DATA = ")
    json.dump(metro_data, f)
    f.write(";\n")

total_size_kb = data_js_path.stat().st_size / 1024
print(f"Successfully generated data.js ({total_size_kb:.1f} KB)")
print("Export complete!")
