#!/usr/bin/env python3
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import memory_profiler
import matplotlib.pyplot as plt

from compsyn.vectors import Vector
from compsyn.logger import get_logger

def get_parser() -> argparse.ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="create a dataset JSON file from a downloads directory")
    parser.add_argument("--downloads", type=Path, help="Path to downloads directory")
    parser.add_argument("--name", type=str, help="Name of this dataset")
    parser.add_argument("--terms", nargs="+", help="terms to create vectors for")
    parser.add_argument("--max", type=int, required=False, help="max number of vectors to load")

    return parser

def directory_with_raw_images(start_path: Path) -> Path:
    """Find the directory nested under path that contains more than 1 filehandle."""
    paths = [ p for p in start_path.iterdir() ]
    if len(paths) == 1 and paths[0].is_dir():
        # If this is another path with nothing but a single directory, keep going
        return directory_with_raw_images(paths[0])
    else:
        # This directory has more than 1 member, it is our answer
        return start_path


if __name__ == "__main__":

    args = get_parser().parse_args()


    if args.create:
        log = get_logger("dataset.create")
        log.info(f"Creating a dataset of vectors called {args.name}")
        output_path = args.downloads.joinpath(f"compsyn-dataset-{args.name}").with_suffix(".json")
        dataset = list() 
        start_mem = memory_profiler.memory_usage()
        start_time = time.time()
        mem_mb_usage_profile = defaultdict(float)
        for term_images_path in [ path for path in args.downloads.iterdir() if path.is_dir()]:
            term = term_images_path.name
            if args.terms is not None:
                if term not in args.terms:
                    # skip terms not in the requested set
                    continue
            if args.max is not None:
                if len(dataset) >= args.max:
                    break
            vector = Vector(term).load_from_folder(directory_with_raw_images(term_images_path), label=term)
            vector_data = vector.to_dict()
            # for more compact storage, we do not store all of the vector data
            for delete_field in ["jzazbz_vector", "rgb_vector", "rgb_ratio", "colorgram_vector"]:
                try:
                    del vector_data[delete_field]
                except KeyError as e:
                    log.warning(f"tried to delete {delete_field} but vector.to_dict() did not have that key")

            vector_data["experiment_name"] = args.name
            dataset.append(vector_data)
            log.debug(f"loaded vector for '{term_images_path.name}'")
            del vector # to try to keep memory usage down over large batches
            if len(dataset) % 10 == 0:
                mem_mb_usage_profile[len(dataset)] = (memory_profiler.memory_usage()[0] - start_mem[0])

        end_time = time.time()
        mem_mb_usage_profile[len(dataset)] = (memory_profiler.memory_usage()[0] - start_mem[0])
        output_path.write_text(json.dumps(dataset))
        plt.plot(list(mem_mb_usage_profile.keys()), list(mem_mb_usage_profile.values()))
        plt.ylabel("Memory Usage (MB)")
        plt.xlabel("Vectors Loaded")
        plt.savefig(f"{args.name}.png")
        log.info(f"{len(dataset)} vectors created in {int(end_time - start_time)/60:.2f} minutes: {output_path}")
