import os
import subprocess
import gzip

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr


def get_url():
    load_dotenv()
    gsmap_user = os.getenv("GSMAP_USER")
    gsmap_pass = os.getenv("GSMAP_PASS")

    return (
        f"ftp://{gsmap_user}:{gsmap_pass}@hokusai.eorc.jaxa.jp/realtime_ver/v7/hourly_G"
    )


def download(start_date=None, end_date=None, freq="h", tz="UTC", out_dir=Path("data/gsmap/raw")):
    base_url = get_url()
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_file = Path("./.gsmap_dl.txt")
    if dl_file.exists():
        dl_file.unlink()

    with open(dl_file, "w") as f:
        dts = pd.date_range(start_date, end_date, freq=freq, tz=tz)
        for dt in dts:
            dt_utc = dt.tz_convert("UTC")
            dt_str = dt_utc.strftime("%Y%m%d.%H00")
            subdir = f"{dt_utc:%Y/%m/%d}"
            fname = f"gsmap_gauge.{dt_str}.dat.gz"
            f.write(f"{base_url}/{subdir}/{fname}\n")

    command = [
        "aria2c",
        f"-i {dl_file}",
        f"-d {out_dir}",
        "-c",
        "--file-allocation=none",
    ]
    try:
        subprocess.run(command)
        print("Done downloading files.")
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
    finally:
        dl_file.unlink()


def to_dataarray(gz_file: Path, var_name: str):
    gz = gzip.GzipFile(gz_file, "rb")
    dd = np.frombuffer(gz.read(), dtype=np.float32)
    dat = dd.reshape((1200, 3600))
    lons = np.linspace(0.05, 359.95, 3600)
    lats = np.linspace(59.95, -59.95, 1200)

    da = xr.DataArray(
         dat,
        coords={"lat": ("lat", lats), "lon": ("lon", lons)},
        name=var_name,
    )
    return da.reindex(lat=da.lat[::-1])


def to_netcdf(in_dir=Path("data/gsmap/raw"), out_dir=Path("data/gsmap")):
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_files = list(in_dir.glob("*.dat.gz"))
    in_files.sort()

    ts = pd.to_datetime("-".join(in_files[0].name.split(".")[1:3]), format="%Y%m%d-%H00", utc=True)
    times = pd.Index(range(len(in_files)), name="time")
    
    das = [to_dataarray(f, "precip") for f in tqdm(in_files, desc="Converting to DataArray...")]
    ds = xr.concat(das, times).to_dataset()

    ds.attrs["Conventions"] = "CF-1.7"
    ds.attrs["source"] = "gsmap_gauge"
    ds.precip.attrs["long_name"] = "hourly averaged rain rate [mm/hr]"
    ds.precip.attrs["missing_value"] = -99.0
    ds.time.attrs["units"] = f"hours since {ts:%Y-%m-%d %H:00:00}"
    ds.lon.attrs["units"] = "degrees_east"
    ds.lon.attrs["standard_name"] = "longitude"
    ds.lon.attrs["long_name"] = "longitude"
    ds.lat.attrs["units"] = "degrees_north"
    ds.lat.attrs["standard_name"] = "latitude"
    ds.lat.attrs["long_name"] = "latitude"
    
    print("Saving as NetCDF...")
    out_file = out_dir / f"gsmap_{ts:%Y-%m-%d_%H}.nc"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds2 = ds.sel(lon=slice(115, 129), lat=slice(4.8, 21))
    ds2.to_netcdf(out_file)


if __name__ == "__main__":
    # download("2024-10-21T00:00:00", "2024-10-24T00:00:00")
    # download("2024-10-21T00:00:00", "2024-10-24T00:00:00", tz="Asia/Manila")
    # download("2024-10-21T00:00:00", "2024-10-21T02:00:00")
    to_netcdf()
