#!/usr/bin/env python3

from collections import Counter, defaultdict
from enum import IntEnum
from tifffile.geodb import GeoKeys
from typing import Iterable, Tuple, List
import laspy
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pathlib
import pyproj
import sys
import threading
import tifffile
import tifffile.geodb
import tqdm

class ReturnCodes(IntEnum):
    SUCCESS = 1
    IGNORED = 2
    FAILED = 3

# Tile map code was made possible thanks to
# https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/
# and contributions to your PBS station from viewers like you, thank you :)
TILE_SIZE = 256
INITIAL_RES = 2 * math.pi * 6378137 / TILE_SIZE
# 156543.03392804062 for tileSize 256 pixels
ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0
# 20037508.342789244
BASE_ZOOM = 17

progress_queue = multiprocessing.Queue()


def resolution(zoom: int) -> float:
    """Resolution (meters/pixel) for given zoom level (measured at Equator)"""
    return INITIAL_RES / (2**zoom)


def meters_to_pixels(mx: int, my: int, zoom: int) -> Tuple[float, float]:
    # Converts EPSG:900913 to pyramid pixel coordinates in given zoom level
    res = resolution(zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (ORIGIN_SHIFT - my) / res
    return px, py


def get_super_tile(
    tz: int, tys: Iterable[int], txs: Iterable[int], dir: pathlib.Path
) -> np.array:
    """Creates a combined array from the desired tiles."""
    super_tile = np.zeros((TILE_SIZE * len(tys), TILE_SIZE * len(txs)))

    for y, ty in enumerate(tys):
        for x, tx in enumerate(txs):
            try:
                super_tile[
                    y * TILE_SIZE : (y + 1) * TILE_SIZE,
                    x * TILE_SIZE : (x + 1) * TILE_SIZE,
                ] = tifffile.imread(dir / str(tz) / str(ty) / f"{tx}.tiff")
            except FileNotFoundError:
                pass

    return super_tile


def get_slope(tile: np.array, zoom: int) -> np.array:
    """
    Gets the slopes in degrees of a tile, comensating for pixel sample
    distance at the particular zoom level.
    """
    dx, dy = np.gradient(tile, resolution(zoom))
    slope = np.sqrt(dx**2 + dy**2)
    slope_deg = np.degrees(np.arctan(slope))
    return slope_deg


def get_zyx(tiff_file: pathlib.Path) -> Tuple[int, int, int]:
    """Gets z,y,x values for a tile file."""
    *_, z, y, x = tiff_file.with_suffix("").parts
    return int(z), int(y), int(x)


def progress_tracker():
    """
    Thread that received updates from the laz worker processes. tqdm needs to
    have all operations performed in a single process. Not sure if this is bug
    free, but it works.
    """
    progress_map = {}
    while update := progress_queue.get():
        file, delta, total = update
        if file not in progress_map:
            if len(file) > 50:
                desc = f"...{file[-47:]}"
            else:
                desc = file
            progress_map[file] = tqdm.tqdm(desc=desc, total=total, unit="points")
        progress_map[file].update(delta)
        if delta == 0:
            progress_map[file].close()


def get_las_projection(las: laspy.LasReader):
    """EXtracts the projection used for the points in the las file"""
    records = {v.record_id: v for v in las.header.vlrs}

    for v in las.header.vlrs:
        # Best scenario, they just define it perfectly for us. <3
        if isinstance(v, laspy.vlrs.known.WktCoordinateSystemVlr):
            return pyproj.Proj(v.string)

        # This might more
        if isinstance(v, laspy.vlrs.known.GeoKeyDirectoryVlr):
            for g in v.geo_keys:
                # Cool id for location of text that describes what we want
                if GeoKeys(g.id) == GeoKeys.PCSCitationGeoKey:
                    raw_data: bytes = records[g.tiff_tag_location].rd
                    piece = raw_data[g.value_offset : g.value_offset + g.count].strip(
                        b"|"
                    )
                    crs = pyproj.CRS.from_user_input(piece.decode())
                    return pyproj.Proj(crs)


def process_laz_to_tiles(file_path: pathlib.Path):
    """
    Worker function that processes all points of a laz file.
    Converts points into heightmaps using the google maps tiles system.
    """
    try:
        las_file = laspy.open(file_path)
        las_points = las_file.read()
    except:
        # print(f"Exception reading {file_path}.")
        return (file_path, {}, ReturnCodes.FAILED)

    las_proj = get_las_projection(las_file)
    if not las_proj:
        raise ValueError(f"Could not find projection for {file_path}")

    las_to_meter = pyproj.Transformer.from_pipeline(
        f"""
    +proj=pipeline
    +step +inv {las_proj.to_proj4()}
    +step +proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs
    """
    )

    tiles = defaultdict(lambda: np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32))

    # This creates a new array so we only want to access .xyz once
    # We will then chunk it afterwards to measure progress
    xyz_points = las_points.xyz

    CHUNK_SIZE = 8192
    total = las_file.header.point_count

    # Test if the data is good enough
    test_points = []
    for mx, my, mz in np.array(las_to_meter.transform(*(xyz_points[:50000].T))).T:
        px, py = meters_to_pixels(mx, my, BASE_ZOOM)
        test_points.append((int(px), int(py)))

    # How many lidar points per pixel
    points = Counter((x, y) for x, y in test_points)
    # How many pixels have a certain number of lidar points
    pixels = Counter(points.values())
    if pixels[1] / sum(pixels.values()) > 0.05:
        # print(f"Ignoring {file_path} because the data is too sparse.")
        return (file_path, {}, ReturnCodes.IGNORED)

    for i in range(0, total, CHUNK_SIZE):
        xs, ys, zs = xyz_points[i : i + CHUNK_SIZE].T

        # First we transform from the las file's coordinate system
        # to lat long and then to meters.
        for mx, my, mz in np.array(las_to_meter.transform(xs, ys, zs)).T:
            # Then based on the tile map zoom, we take the meter coordinate and
            # convert it to pixel coordinates on entire map.
            px, py = meters_to_pixels(mx, my, BASE_ZOOM)
            # We floor the pixel coordinate and then divmod to get the tile
            # coordinate and pixel location within the tile.
            tx, px = divmod(int(px), TILE_SIZE)
            ty, py = divmod(int(py), TILE_SIZE)

            # this might be slow
            t = tiles[(tx, ty)]
            t[py][px] = max(t[py][px], mz)

        progress_queue.put((str(file_path), len(xs), total))
    progress_queue.put((str(file_path), 0, total))

    return (file_path, dict(tiles), ReturnCodes.SUCCESS)


class TileMaker:
    def __init__(self, costmap_dir: pathlib.Path):
        self.laz_dir = costmap_dir / "laz"
        self.height_dir = costmap_dir / "heights"
        self.slopes_dir = costmap_dir / "slopes"
        self.tiles_dir = costmap_dir / "tiles"
        self.done_laz = costmap_dir / "done_laz.txt"
        self.failed_laz = costmap_dir / "failed_laz.txt"
        self.ignored_laz = costmap_dir / "ignored_laz.txt"

    def get_done_laz(self) -> List[str]:
        """
        Get the list of laz files we have already processed. Stored as a
        newline seperated text file of just the laz filenames.
        """
        self.done_laz.touch()
        with self.done_laz.open("r") as f:
            return f.read().splitlines()

    def add_done_laz(self, laz_file: pathlib.Path):
        """Adds a laz file to the list of processed files."""
        with self.done_laz.open("a+") as f:
            f.write(laz_file.name + "\n")

    def add_failed_laz(self, laz_file: pathlib.Path):
        """Adds a laz file to the list of failed files."""
        with self.failed_laz.open("a+") as f:
            f.write(laz_file.name + "\n")

    def add_ignored_laz(self, laz_file: pathlib.Path):
        """Adds a laz file to the list of ignored files."""
        with self.ignored_laz.open("a+") as f:
            f.write(laz_file.name + "\n")

    def height_file(self, z: int, y: int, x: int) -> pathlib.Path:
        return self.height_dir / str(z) / str(y) / f"{x}.tiff"

    def slopes_file(self, z: int, y: int, x: int) -> pathlib.Path:
        return self.slopes_dir / str(z) / str(y) / f"{x}.tiff"

    def tiles_file(self, z: int, y: int, x: int) -> pathlib.Path:
        return self.tiles_dir / str(z) / str(y) / f"{x}.png"

    def convert_all_lazs_to_tiff(self):
        """
        Finds all laz files that have not been processed previously. Makes a
        process pool to utilize all threads for laz processing. Processes and
        threads are not the same and that means we have to do funny things to
        make the progress bars show up nicely.
        """
        lazs = {f.name for f in self.laz_dir.glob("*.laz")}
        done_lazs = set(self.get_done_laz())
        todo = [self.laz_dir / f for f in (lazs - done_lazs)]

        pool = multiprocessing.Pool()
        for file_path, tiles, return_code in pool.imap_unordered(process_laz_to_tiles, todo):
            if return_code == ReturnCodes.FAILED:
                self.add_failed_laz(file_path)
            elif return_code == ReturnCodes.IGNORED:
                self.add_ignored_laz(file_path)
            elif return_code == ReturnCodes.SUCCESS:
                for (tx, ty), tile in tiles.items():
                    output_name = self.height_file(BASE_ZOOM, ty, tx)
                    output_name.parent.mkdir(parents=True, exist_ok=True)
                    # Gotta merge if the file was already there
                    if output_name.exists():
                        old_tile = tifffile.imread(output_name)
                        # Merge by taking the maximum value at each point
                        np.maximum(tile, old_tile, out=tile)

                    tifffile.imwrite(str(output_name), tile)

                # Record that we've processed this laz file
                self.add_done_laz(file_path)

    def convert_height_to_slope(self):
        """
        Converts all the height tiffs to slope. Grabs surrounding tiles so
        that slopes calculated at boundaries are consistent across tiles.
        """
        for tile_path in self.height_dir.glob("*/*/*.tiff"):
            z, y, x = get_zyx(tile_path)

            output_name = self.slopes_file(z, y, x)
            if output_name.exists():
                continue

            print(output_name)
            combined = get_slope(
                get_super_tile(
                    z, (y - 1, y, y + 1), (x - 1, x, x + 1), self.height_dir
                ),
                z,
            )
            middle = combined[TILE_SIZE : 2 * TILE_SIZE, TILE_SIZE : 2 * TILE_SIZE]

            output_name.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(output_name), middle)

    def merge_tiff_for_layers(self):
        """
        Starting from the most zoomed in layer we generated the height maps
        with, merge and downsample groups of four tiles to generate the next
        layer up.
        """
        for z in range(BASE_ZOOM - 1, 0 - 1, -1):
            for tile_path in self.slopes_dir.glob(f"{z+1}/*/*.tiff"):
                _, y, x = get_zyx(tile_path)
                ty = y // 2
                tx = x // 2

                output_name = self.slopes_file(z, ty, tx)
                if output_name.exists():
                    continue
                print(output_name)

                combined = get_super_tile(
                    z + 1, (ty * 2, ty * 2 + 1), (tx * 2, tx * 2 + 1), self.slopes_dir
                )
                shrunk = combined.reshape(TILE_SIZE, 2, TILE_SIZE, 2).max(3).max(1)

                output_name.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(output_name, shrunk)

    def convert_tiffs_to_png(self):
        """
        Finally convert all the slope tiffs into tile pngs that can be used by
        leaflet. Here we can adjust the colormap used for the image generation.
        """
        slopes = list(self.slopes_dir.glob("*/*/*.tiff"))
        slopes.sort(key=get_zyx, reverse=True)
        for tile_path in slopes:
            z, y, x = get_zyx(tile_path)
            output_name = self.tiles_file(z, y, x)
            if output_name.exists():
                continue
            print(output_name)

            tile = tifffile.imread(tile_path)
            output_name.parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(output_name, tile, cmap=plt.get_cmap("turbo"), vmin=0, vmax=80)


def main(laz_dir: pathlib.Path):
    threading.Thread(target=progress_tracker, daemon=True).start()
    tiler = TileMaker(laz_dir)
    # Step 1. Process laz files into tiff heightmaps tiles Z=17
    tiler.convert_all_lazs_to_tiff()
    # Step 2. Convert all height data into slope
    tiler.convert_height_to_slope()
    # Step 3. Combine and downsample to make slope tiles for lower Z values
    tiler.merge_tiff_for_layers()
    # Step 4. Convert all slope tiffs to png
    tiler.convert_tiffs_to_png()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} DIRECTORY")
        print()
        print("Where DIRECTORY/laz is a directory containing a collection of laz files.")
        print("Files will be output in DIRECTORY/<heights, slopes, tiles>.")
        print("Use ./download_helper.py to determine what laz files you should download.")
        print("I recommend using a download manager like uget to download all the files successfully.")
        exit()

    costmap_dir = pathlib.Path(sys.argv[1])
    if not costmap_dir.exists() or not costmap_dir.is_dir():
        print(f"{costmap_dir} is not a valid directory.")
        exit(1)

    main(costmap_dir)
