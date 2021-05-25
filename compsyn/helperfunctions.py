#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import random
import requests
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from google.cloud import vision_v1p2beta1 as vision

from .logger import get_logger
from .utils import env_default


def get_google_application_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:

    if parser is None:
        parser = argparse.ArgumentParser()

    google_vision_parser = parser.add_argument_group("google_vision")

    google_vision_parser.add_argument(
        "--google-application-credentials",
        type=str,
        action=env_default("COMPSYN_GOOGLE_APPLICATION_CREDENTIALS"),
        required=False,
        help="Credentials file for accessing GCloud services like Google Vision API, etc.",
    )

    return parser


def run_google_vision(img_urls_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """
       Use the Google vision API to return a set of classification labels for each image collected from 
       Google using the search_and_download function. Each label assigned by Google vision is associated 
       with a score indicating Google's confidence in the fit fo the label for the image.
       
       img_urls_dict: dictionary containing image_urls
    """

    log = get_logger("run_google_vision")

    log.info("Classifying Imgs. w. Google Vision API...")

    # copy environment variable from configuration to the name where google API expects to find it
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
        "COMPSYN_GOOGLE_APPLICATION_CREDENTIALS"
    )

    client = vision.ImageAnnotatorClient()
    image = vision.types.Image()

    for search_term in img_urls_dict.keys():
        img_urls = img_urls_dict[search_term]
        img_classified_dict = {}
        img_classified_dict[search_term] = {}
        log.info(f"Classifying {len(img_urls)} images for {search_term}")

        for image_uri in img_urls:
            try:
                image.source.image_uri = image_uri
                response = client.label_detection(image=image)
                img_classified_dict[image_uri] = {}

                for label in response.label_annotations:
                    img_classified_dict[search_term][image_uri] = {}
                    img_classified_dict[search_term][image_uri][
                        label.description
                    ] = label.score

            except:
                pass

    return img_classified_dict


def write_to_json(to_save: Dict[str, Any], filename: str) -> None:
    """ write dictionary to existing json file"""
    with open(filename, "w") as to_write_to:
        json.dump(to_save, to_write_to, indent=4)


def write_img_classifications_to_file(
    work_dir: Union[str, Path],
    search_terms: List[str],
    img_classified_dict: Dict[str, Any],
) -> None:
    """
       Store Google vision's classifications for images in a json file, which can then be retrieved for 
       the purposes of filtering and also statistical analyses.  
       
       search_terms: terms used for querying Google
       img_classified_dict: dictionary of image URLs and classifications from Google Vision
    """

    log = get_logger("write_img_classifications_to_file")

    base_dir = Path(work_dir).joinpath("image_classifications")
    base_dir.mkdir(exist_ok=True, parents=True)

    for term in search_terms:
        term_data = img_classified_dict[term]

        if term_data:
            filename = base_dir.joinpath(
                "classifications_"
                + term
                + "_"
                + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
                + ".json"
            )

            if filename.is_file():
                log.info("File already exists! Appending to file.. ")

                term_data_orig = json.loads(filename.read_text())
                term_data_orig.update(term_data)
                filename.write_text(json.dumps(term_data_orig))

            else:
                log.info("File new! Saving..")
                filename.write_text(json.dumps(term_data, indent=2))
