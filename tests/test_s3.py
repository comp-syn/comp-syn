from __future__ import annotations
import time
from pathlib import Path

import pytest

from compsyn.s3 import (
    get_s3_args,
    get_s3_client,
    s3_object_exists,
    list_object_paths_in_s3,
    upload_file_to_s3,
    download_file_from_s3,
)

# Must have these S3 environment variables set for these tests to work:
# COMPSYN_S3_BUCKET, COMPSYN_S3_ACCESS_KEY_ID, COMPSYN_S3_SECRET_ACCESS_KEY
#

LOCAL_PATH = Path(__file__).parent.joinpath("test-assets/compress-image-ratio.png")
S3_PATH = Path(f"pytest/test-assets/compress-image-ratio-{int(time.time())}.png")


@pytest.mark.credentials
def test_get_s3_client() -> None:
    s3_args, unknown = get_s3_args().parse_known_args()
    get_s3_client(s3_args)


@pytest.mark.credentials
def test_upload_file_to_s3() -> None:
    upload_file_to_s3(local_path=LOCAL_PATH, s3_path=S3_PATH)
    upload_file_to_s3(local_path=LOCAL_PATH, s3_path=S3_PATH)
    upload_file_to_s3(local_path=LOCAL_PATH, s3_path=S3_PATH, overwrite=True)


@pytest.mark.online
@pytest.mark.depends(on=["test_upload_file_to_s3"])
def test_s3_object_exists() -> None:
    assert s3_object_exists(S3_PATH)


@pytest.mark.online
@pytest.mark.depends(on=["test_upload_file_to_s3"])
def test_s3_list_object_paths_in_s3() -> None:
    count = 0
    for s3_path in list_object_paths_in_s3(S3_PATH.parent):
        count += 1
    assert count > 0


@pytest.mark.online
@pytest.mark.depends(on=["test_upload_file_to_s3"])
def test_download_file_from_s3() -> None:
    tmp_local_path = LOCAL_PATH.with_suffix(".pytest.tmp")
    assert not tmp_local_path.is_file()
    download_file_from_s3(local_path=tmp_local_path, s3_path=S3_PATH)
    assert tmp_local_path.is_file()
    download_file_from_s3(local_path=tmp_local_path, s3_path=S3_PATH)
    download_file_from_s3(local_path=tmp_local_path, s3_path=S3_PATH, overwrite=True)
    tmp_local_path.unlink()
    assert not tmp_local_path.is_file()
