"""Environment smoke test.

Run before anything else on a fresh machine:

    python -m src.check_env
"""

import importlib
import sys

REQUIRED = [
    "pandas", "numpy", "geopandas", "shapely", "pyproj", "rasterio",
    "rasterstats", "osmnx", "networkx", "sklearn", "statsmodels",
    "libpysal", "esda", "shap", "mapclassify", "matplotlib",
]

OPTIONAL = ["contextily", "spreg", "pyarrow", "requests"]


def check(names, label):
    ok, bad = [], []
    for name in names:
        try:
            mod = importlib.import_module(name)
            ok.append((name, getattr(mod, "__version__", "unknown")))
        except Exception as exc:  # noqa: BLE001 - want the reason, whatever it is
            bad.append((name, f"{type(exc).__name__}: {exc}"))

    print(f"\n{label}")
    print("-" * len(label))
    for name, ver in ok:
        print(f"  ok    {name:<14} {ver}")
    for name, err in bad:
        print(f"  FAIL  {name:<14} {err}")
    return bad


def main():
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")

    missing = check(REQUIRED, "Required")
    check(OPTIONAL, "Optional")

    # A real geometry round-trip catches broken GEOS/PROJ installs that a bare
    # import will happily pass.
    print("\nGeometry round-trip")
    print("-" * len("Geometry round-trip"))
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {"name": ["Parry's Corner"]},
            geometry=[Point(80.2874, 13.0937)],
            crs="EPSG:4326",
        )
        projected = gdf.to_crs("EPSG:32644")
        area_km2 = projected.buffer(800).area.iloc[0] / 1e6
        print(f"  ok    reprojected to UTM 44N, 800 m buffer = {area_km2:.3f} km2 (expect ~2.011)")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        missing.append(("geometry round-trip", str(exc)))

    if missing:
        print(f"\n{len(missing)} problem(s). Fix with: python -m pip install -r requirements.txt")
        return 1
    print("\nEnvironment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
