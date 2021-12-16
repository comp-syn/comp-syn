from __future__ import annotations

"""
S3 is a common protocol for storing data. Many cloud providers offer an S3-compatible remote storage service.

Compsyn can be configured to optionally store data in S3.

"""

import argparse
import os
from pathlib import Path

import boto3

from .logger import get_logger
from .utils import env_default


class S3Error(Exception):
    pass


class NoObjectInS3Error(S3Error):
    pass


def get_s3_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """ Fetches arguments for S3 from the environment """

    if parser is None:
        parser = argparse.ArgumentParser()

    s3_parser = parser.add_argument_group("s3")

    s3_parser.add_argument(
        "--s3-bucket",
        type=str,
        action=env_default("COMPSYN_S3_BUCKET"),
        required=False,
        help="bucket where img data is stored in S3",
    )
    s3_parser.add_argument(
        "--s3-region-name",
        type=str,
        required=False,
        action=env_default("COMPSYN_S3_REGION_NAME"),
        help="S3 region",
    )
    s3_parser.add_argument(
        "--s3-endpoint-url",
        action=env_default("COMPSYN_S3_ENDPOINT_URL"),
        required=False,
        help="S3 endpoint URL (only required for non-AWS S3)",
    )
    s3_parser.add_argument(
        "--s3-access-key-id",
        type=str,
        action=env_default("COMPSYN_S3_ACCESS_KEY_ID"),
        required=False,
    )
    s3_parser.add_argument(
        "--s3-secret-access-key",
        type=str,
        action=env_default("COMPSYN_S3_SECRET_ACCESS_KEY"),
        required=False,
    )

    return parser


def get_s3_client(args: argparse.Namespace) -> botocore.clients.s3:
    """ 
        Convenience method for validating arguments and instantiating a boto s3 client object.
        Since using S3 in compsyn is optional, we hold off on argument validation until a client is actually requested 
    """

    assert args.s3_region_name is not None, "set COMPSYN_S3_REGION_NAME"
    assert args.s3_access_key_id is not None, "set COMPSYN_S3_ACCESS_KEY_ID"
    assert args.s3_secret_access_key is not None, "set COMPSYN_S3_SECRET_ACCESS_KEY"
    assert args.s3_bucket is not None, "set COMPSYN_S3_BUCKET"

    return boto3.session.Session().client(
        "s3",
        region_name=args.s3_region_name,
        endpoint_url=args.s3_endpoint_url,
        aws_access_key_id=args.s3_access_key_id,
        aws_secret_access_key=args.s3_secret_access_key,
    )


def s3_object_exists(s3_path: Path) -> bool:
    """ Check whether a given path in S3 exists """

    s3_args, unknown = get_s3_args().parse_known_args()
    s3_client = get_s3_client(s3_args)
    log = get_logger("s3_object_exists")

    try:
        s3_client.get_object(Bucket=s3_args.s3_bucket, Key=str(s3_path))
        return True
    except s3_client.exceptions.NoSuchKey:
        return False


class NoS3DataError(Exception):
    pass


def list_object_paths_in_s3(s3_prefix: Path) -> Generator[Path]:
    """ Paginate contents at an S3 prefix, yielding Paths to S3 objects """

    s3_args, unknown = get_s3_args().parse_known_args()
    s3_client = get_s3_client(s3_args)
    log = get_logger("list_object_paths_in_s3")

    resp = s3_client.list_objects_v2(Bucket=s3_args.s3_bucket, Prefix=str(s3_prefix))

    if "Contents" not in resp:
        raise NoS3DataError(f"No data at prefix {s3_prefix}")

    while True:
        yield from (Path(obj["Key"]) for obj in resp["Contents"])

        if resp["IsTruncated"]:
            continuation_key = resp["NextContinuationToken"]
            resp = s3_client.list_objects_v2(
                Bucket=s3_args.s3_bucket,
                Prefix=str(s3_prefix),
                ContinuationToken=continuation_key,
            )
        else:
            break


def upload_file_to_s3(local_path: Path, s3_path: Path, overwrite: bool = False) -> None:
    """ By default, existing S3 objects are not overwritten """
    import warnings

    warnings.filterwarnings(
        action="ignore", message="unclosed", category=ResourceWarning
    )

    s3_args, unknown = get_s3_args().parse_known_args()
    s3_client = get_s3_client(s3_args)
    log = get_logger("upload_file_to_s3")

    try:
        # only write files to s3 that don't already exist unless overwrite is passed
        if s3_object_exists(s3_path) and not overwrite:
            log.debug(
                f"s3://{s3_args.s3_bucket}/{s3_path} already exists in s3, not overwriting"
            )
            return

        s3_client.put_object(
            Body=local_path.read_bytes(), Bucket=s3_args.s3_bucket, Key=str(s3_path)
        )
        log.debug(f"uploaded s3://{s3_args.s3_bucket}/{s3_path}")

    except s3_client.exceptions.ClientError as exc:
        # catch and raise any errors generated while attempting to communicate with s3
        s3_client_attributes = {
            attr: getattr(s3_client, attr) for attr in s3_client.__dict__.keys()
        }
        s3_client_attributes.update(
            {"bucket": s3_args.s3_bucket, "object_path": s3_path,}
        )
        raise S3Error(f"{s3_client_attributes} S3 ClientError") from exc


def download_file_from_s3(
    local_path: Path, s3_path: Path, overwrite: bool = False
) -> None:
    """ By default, local Paths are not overwritten """

    import warnings

    warnings.filterwarnings(
        action="ignore", message="unclosed", category=ResourceWarning
    )

    s3_args, unknown = get_s3_args().parse_known_args()
    s3_client = get_s3_client(s3_args)
    log = get_logger("download_file_from_s3")

    try:
        # only download files from s3 that don't already exist locally unless overwrite is passed
        if local_path.is_file():
            if not overwrite:
                log.debug(f"{local_path} already exists locally, not overwriting")
                return

        s3_obj = s3_client.get_object(Bucket=s3_args.s3_bucket, Key=str(s3_path))
        local_path.parent.mkdir(exist_ok=True, parents=True)
        local_path.write_bytes(s3_obj["Body"].read())
        log.debug(f"downloaded {local_path} from s3")

    except s3_client.exceptions.ClientError:
        # catch and raise any errors generated while attempting to communicate with s3
        s3_client_attributes = {
            attr: getattr(s3_client, attr) for attr in s3_client.__dict__.keys()
        }
        s3_client_attributes.update(
            {"bucket": bucket, "object_path": object_path,}
        )
        raise S3Error(f"{s3_client_attributes} S3 ClientError")
