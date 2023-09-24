# USGS Scripts

## 1. Download laz files

### Option 1. Use [download_helper.py](download_helper.py) and [download_merger.py](download_merger.py)
Helper script that automates option #2 and give you a list of files to download that cover a rectange centered at a gps point. Run it for more help. Use the download_merger to merge several URL files and remove redundant URLs.

### Option 2. https://apps.nationalmap.gov/downloader/
Select "Elevation Source Data (3DEP) - Lidar, IfSAR" -> "Lidar Point Cloud (LPC)" -> "LAS,LAZ". Draw the region you want to download. Download the laz files.

## 2. [lidar_tile_maker.py](lidar_tile_maker.py)
Once you have downloaded all of your laz files. Put them into a folder and then call `./lidar_tile_maker.py FOLDER` and away you go. It will spawn as many processes as your computer has threads and will go through the files as fast as it can. Do not run multiple instances of this script at once.

It will first generate heightmaps tiffs stored in `FOLDER/heights` and then compute slopes for all the tile layers in `FOLDER/slopes`. Finally it will convert those slopes into png tiles in `FOLDER/tiles` that can be used with leaflet (or pyqtlet in our case).
