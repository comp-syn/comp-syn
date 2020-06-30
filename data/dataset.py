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
    parser = argparse.ArgumentParser(prog="compsyn-datasets")

    sub_parsers = parser.add_subparsers(dest="command")
    create_parser = sub_parsers.add_parser(
        "create", help="create a dataset JSON file from a downloads directory"
    )
    create_parser.add_argument("--name", type=str, help="Name of this dataset")
    create_parser.add_argument(
        "--downloads", type=Path, help="Path to downloads directory"
    )
    create_parser.add_argument(
        "--output-path",
        type=Path,
        required=False,
        help="Path to write JSON file to, optional. Default behaviour is to write a file in the --downloads folder.",
    )
    create_parser.add_argument("--terms", nargs="+", help="terms to create vectors for")
    create_parser.add_argument(
        "--max", type=int, required=False, help="max number of vectors to load"
    )
    create_parser.add_argument(
        "--include", nargs="+", help="fields to include for each vector"
    )
    create_parser.add_argument(
        "--profile",
        action="store_true",
        help="Create a memory profile graph to track memory usage as a function of vectors loaded",
    )

    return parser


def directory_with_raw_images(start_path: Path) -> Path:
    """Find the directory nested under path that contains more than 1 filehandle."""
    paths = [p for p in start_path.iterdir()]
    if len(paths) == 1 and paths[0].is_dir():
        # If this is another path with nothing but a single directory, keep going
        return directory_with_raw_images(paths[0])
    else:
        # This directory has more than 1 member, it is our answer
        return start_path


if __name__ == "__main__":

    args = get_parser().parse_args()

    if args.command == "create":
        log = get_logger("dataset.create")
        log.info(f"Creating a dataset of vectors called {args.name}")
        output_path = (
            args.downloads.joinpath(f"compsyn-dataset-{args.name}").with_suffix(".json")
            if args.output_path is None
            else args.output_path.with_suffix(".json")
        )
        dataset = list()
        start_mem = memory_profiler.memory_usage()
        start_time = time.time()
        mem_mb_usage_profile = defaultdict(float)
        for term_images_path in [
            path for path in args.downloads.iterdir() if path.is_dir()
        ]:
            term = term_images_path.name
            if args.terms is not None:
                if term not in args.terms:
                    # skip terms not in the requested set
                    continue
            if args.max is not None:
                if len(dataset) >= args.max:
                    break
            vector = Vector(term).load_from_folder(
                directory_with_raw_images(term_images_path), label=term
            )
            vector_data = vector.to_dict()

            # delete all fields not explicitly requested
            for field in list(vector_data.keys()):
                if field not in args.include:
                    log.debug(f"dropped field {term_images_path.name}")
                    del vector_data[field]

            vector_data["experiment_name"] = args.name
            dataset.append(vector_data)
            log.debug(f"loaded vector for '{term_images_path.name}'")

            del vector  # to try to keep memory usage down over large batches

            if len(dataset) % 10 == 0:
                mem_mb_usage_profile[len(dataset)] = (
                    memory_profiler.memory_usage()[0] - start_mem[0]
                )

        end_time = time.time()
        end_mem_usage = memory_profiler.memory_usage()[0] - start_mem[0]
        mem_mb_usage_profile[len(dataset)] = end_mem_usage
        output_path.write_text(json.dumps(dataset))
        log.info(
            f"{len(dataset)} vectors created in {int(end_time - start_time)/60:.2f} minutes. {end_mem_usage} MB of memory in use at loop termination."
        )
        if args.profile:
            profile_plot_path = output_path.parent.joinpath(args.name).with_suffix(
                ".png"
            )
            plt.plot(
                list(mem_mb_usage_profile.keys()), list(mem_mb_usage_profile.values())
            )
            plt.ylabel("Memory Usage (MB)")
            plt.xlabel("Vectors Loaded")
            plt.savefig(profile_plot_path)
            print(f"\n\tMEMORY PROFILE: {profile_plot_path.resolve()}")
        print(f"\n\tVECTORS: {output_path.resolve()}\n")
