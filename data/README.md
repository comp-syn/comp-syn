## Create JSON datasets of comp-syn vectors

Currently `dataset.py` has one mode - `create`:
```
$ poetry run ./data/dataset.py create --help
usage: compsyn-datasets create [-h] [--name NAME] [--downloads DOWNLOADS] [--output-path OUTPUT_PATH] [--terms TERMS [TERMS ...]]
                               [--max MAX]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of this dataset
  --downloads DOWNLOADS
                        Path to downloads directory
  --output-path OUTPUT_PATH
                        Path to write JSON file to, optional. Default behaviour is to write a file in the --downloads folder.
  --terms TERMS [TERMS ...]
                        terms to create vectors for
  --max MAX             max number of vectors to load

```

## examples

```
python3 ./data/dataset.py create --downloads /path/to/downloads/folder/ --name what-to-call-this-dataset
```

This will create `compsyn.vectors.Vector` objects out of each term in the downloads folder, and create a list of compact vector representations for each word. Once all of the requested vectors are created, a JSON file is written.

This script exposes some optional methods for generating subsets of data, for instance if you only want to generate vectors for a subset of terms, you can do so like this:

```
python3 ./data/dataset.py create --downloads /path/to/downloads/folder/ --name what-to-call-this-dataset --terms term1 term2 term3 ...
```

You can also optionally set a maximum number of vectors to gather, this is mostly useful for quickly validating what your data will look like before creating large datasets.

```
# create a JSON dump with only a pair of vectors to validate data structure
python3 ./data/dataset.py create --downloads /path/to/downloads/folder/ --name what-to-call-this-dataset --max 2
```

To explore the memory footprint of loading vectors, the `create` mode also has a `--profile` option, which will create a graph of memory usage as a function of vectors loaded:

```
# create a JSON dump with a subset of vectors to explore memory usage
python3 ./data/dataset.py create --downloads /path/to/downloads/folder/ --name what-to-call-this-dataset --max 20 --profile
```
