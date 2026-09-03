# QGIS map package

Open `chennai_mapathon_ridership_desktop.qgz` in QGIS. If double-clicking the
project does not work, run `open_chennai_mapathon_qgis.bat`.

Generated contents:

- `chennai_mapathon_layers_desktop.gpkg` - QGIS-ready GeoPackage in EPSG:32644.
- `Bhuvan LULC 2005-06`, `2011-12`, `2015-16` - from `D:/Mapathon Data Files/`.
- `OpenStreetMap basemap` - visual reference under the thematic layers.
- `Phase 1/Phase 2 metro guide lines` - real OSM/Overpass metro linework.
- `Phase 2 stations by predicted boardings` - station size and colour show predicted daily boardings.
- `Low-confidence station positions` - dashed rings flag uncertain Phase 2 locations.
- `800 m station catchments` - outline-only catchments rebuilt around the aligned map points.

Station points in this QGIS package are cartographically snapped to the real
metro linework. The original source coordinates are retained as `source_x` and
`source_y`; the map movement is recorded in `map_snap_m`, and low-confidence
locations remain visibly flagged.

Exports:

- `../outputs/maps/chennai_mapathon_qgis_preview_desktop.pdf`
- `../outputs/maps/chennai_mapathon_qgis_preview_desktop.png`
- `../outputs/figures/chennai_top15_phase2_predictions.png`

Rebuild with:

```powershell
& 'D:\QGIS\bin\python-qgis.bat' 'D:\testrun\Mapathon\scripts\build_qgis_project.py'
```
