#!/usr/bin/env python3

import sys
import pathlib
import pyproj
import requests


def main(lat_center, lon_center, width, height, output_file):
    center_enu_to_gps = pyproj.Transformer.from_pipeline(
        f"""
    +proj=pipeline
    +step +inv +proj=topocentric +ellps=WGS84 +lat_0={lat_center} +lon_0={lon_center} +h_0=0
    +step +inv +proj=cart +ellps=WGS84
    """
    )

    # find bottom left corner
    lon0, lat0 = center_enu_to_gps.transform(-width // 2, -height // 2)
    # find top right corner
    lon1, lat1 = center_enu_to_gps.transform(width // 2, height // 2)

    print(f"Xmax: {lon1:.6f}, Ymax: {lat1:.6f}")
    print(f"Xmin: {lon0:.6f}, Ymin: {lat0:.6f}")
    print()

    total = 100
    got = 0

    with open(output_file, 'w') as file:

        while got < total:
            params = {
                "prodFormats": "LAS,LAZ",
                "datasets": "Lidar Point Cloud (LPC)",
                "polygon": f"{lon0} {lat0}, {lon1} {lat0}, {lon1} {lat1}, {lon0} {lat1}, {lon0} {lat0}",
                "offset": got,
                "max": 100,
            }

            resp = requests.get(
                "https://tnmaccess.nationalmap.gov/api/v1/products", params=params
            ).json()
            if got == 0:
                print(f'Found {resp["total"]} USGS data files to download')

            got += len(resp["items"])
            total = resp["total"]

            for item in resp["items"]:
                file.write(item["downloadURL"] + '\n')


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} LAT LONG WIDTH_METERS HEIGHT_METERS OUTPUT_FILE")
        print()
        print(
            "I recommend using https://ugetdm.com/ for a good download manager\n"
            "to get the files. Trust me you're going to need it. Some of these\n"
            "are very slow and the download manager opens multiple connections\n"
            "if you let it. It also retries failed downloads where it left off"
        )
        print()
        print(
            "Use the download merger if you are using this script for several\n"
            "locations and need to combine the results without duplicates.\n"
        )
        exit(1)

    lat = float(sys.argv[1])
    long = float(sys.argv[2])
    width = float(sys.argv[3])
    height = float(sys.argv[4])
    output_file = pathlib.Path(sys.argv[5])

    main(lat, long, width, height, output_file)
