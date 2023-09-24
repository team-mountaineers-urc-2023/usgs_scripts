#!/usr/bin/env python3

import sys
import pathlib

def main(input_dir: pathlib.Path, output_file: pathlib.Path):

	# file all url files
	url_files = {f.name for f in input_dir.glob("*")}

	# read and deduplicate urls in all url files
	combined_urls = set()
	url_count = 0
	for url_file_basename in url_files:
		url_file = input_dir / url_file_basename
		with open(url_file, 'r') as file:
			for line in file:
				url = line.strip()
				combined_urls.add(url)
				url_count += 1

	print(f'Found {url_count} USGS data URLs before deduplication')
	print(f'Reduced to {len(combined_urls)} USGS data URLs after deduplication')

	# write resulting urls to output file
	with open(output_file, 'w') as file:
		for url in combined_urls:
			file.write(url + '\n')

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print(f"Usage: {sys.argv[0]} INPUT_DIR OUTPUT_FILE")
		print()
		print(
			"I recommend using https://ugetdm.com/ for a good download manager\n"
			"to get the files. Trust me you're going to need it. Some of these\n"
			"are very slow and the download manager opens multiple connections\n"
			"if you let it. It also retries failed downloads where it left off"
		)
		exit(1)

	input_dir = pathlib.Path(sys.argv[1])
	output_file = pathlib.Path(sys.argv[2])

	main(input_dir, output_file)
