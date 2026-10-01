# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import hashlib
import json
import os
import ssl
import urllib.request
from collections import defaultdict
from pathlib import PurePosixPath
from zipfile import ZipFile

from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

SCENES_PER_CHUNK = 10
REQUIRED_SCENE_FILES = {
    "ase_scene_language.txt",
    "object_instances_to_classes.json",
    "semidense_points.csv.gz",
}
IMAGE_FOLDERS = {"rgb", "depth", "instances"}


# Copied from tqdm website/documentation - probably worth moving into a utils.py
def urllib_tqdm_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


# Handler for list of scene ids. Handles lists of integers and ranges.
def ASEIdsParser(string):
    try:
        ids = string.split(",")
        ids = [
            (
                list(range(int(x.split("-")[0]), int(x.split("-")[1]) + 1))
                if "-" in x
                else [int(x)]
            )
            for x in ids
        ]
        ids = [item for sublist in ids for item in sublist]
        return list(set(ids))
    except Exception as e:
        print("Error: ", e)
        raise argparse.ArgumentTypeError(
            "Scene ids must be comma separated integers or ranges. For example: 1,2,3-5,6"
        )


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered_value = value.lower()
    if lowered_value in {"true", "1", "yes", "y"}:
        return True
    if lowered_value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value for --unzip.")


parser = argparse.ArgumentParser(
    prog="Aria Synthetic Environments Downloader",
    description="Downloads Aria Synthetic Environments as defined in CDN json file.",
)
parser.add_argument(
    "--set",
    help="The type of scenes to download. Options are train or test.",
    choices=["train", "test"],
    required=True,
)
parser.add_argument(
    "--scene-ids",
    help="Scene ids to download.",
    required=True,
    type=ASEIdsParser,
)
parser.add_argument(
    "--cdn-file",
    help="Input file listing the CDN urls, downloaded from ASE website.",
    required=True,
)

parser.add_argument("--output-dir", help="Output directory", required=True)

parser.add_argument(
    "--unzip",
    help="Unzip the downloaded zip files. ",
    required=True,
    type=str2bool,
)


def load_meta_data(cdn_file: str):
    # Load the metadata file downloaded from the ASE website.
    with open(cdn_file) as fp:
        metadata = json.load(fp)
    return metadata


def get_matching_scene_id(member_name: str, requested_scene_ids: set):
    member_path = PurePosixPath(member_name)
    if member_name.endswith("/") or member_path.name not in REQUIRED_SCENE_FILES:
        return None

    path_parts = member_path.parts
    if IMAGE_FOLDERS.intersection(path_parts):
        return None

    for part in reversed(path_parts[:-1]):
        if part in requested_scene_ids:
            return part

    return None


def extract_required_files(
    zip_filename: str, output_dir: str, requested_scene_ids: set
) -> tuple[int, dict]:
    extracted_count = 0
    found_files_by_scene = defaultdict(set)
    with ZipFile(zip_filename, "r") as zip_ref:
        for member in zip_ref.infolist():
            matched_scene_id = get_matching_scene_id(
                member.filename, requested_scene_ids
            )
            if matched_scene_id is not None:
                zip_ref.extract(member, output_dir)
                found_files_by_scene[matched_scene_id].add(
                    PurePosixPath(member.filename).name
                )
                extracted_count += 1
    return extracted_count, dict(found_files_by_scene)


def warn_for_missing_files(
    found_files_by_scene: dict, requested_scene_ids: set, chunk_scene_ids: set
):
    for scene_id in sorted(chunk_scene_ids, key=int):
        missing_files = REQUIRED_SCENE_FILES - found_files_by_scene.get(scene_id, set())
        for missing_file in sorted(missing_files):
            print(
                f"Warning: scene {scene_id} is missing required file {missing_file}"
            )


def main(
    cdn_file: str, output_dir: str, scene_ids: list, set_type: str, unzip_flag: bool
):
    # Load the metadata file downloaded from the ASE website.
    metadata = load_meta_data(cdn_file)

    # Create the output directory.
    if not os.path.exists(output_dir):
        print(f"Creating local output folder {output_dir}")
        os.makedirs(output_dir)

    requested_scene_ids = {str(scene_id) for scene_id in scene_ids}
    chunk_ids_to_download = list(set([x // SCENES_PER_CHUNK for x in scene_ids]))
    chunk_ids_to_download.sort()
    for i, chunk_id in enumerate(chunk_ids_to_download):
        chunk_start_scene_id = chunk_id * SCENES_PER_CHUNK
        chunk_end_scene_id = chunk_start_scene_id + SCENES_PER_CHUNK
        chunk_scene_ids = {
            scene_id
            for scene_id in requested_scene_ids
            if chunk_start_scene_id <= int(scene_id) < chunk_end_scene_id
        }
        chunk_filename = "{}_chunk_{}.zip".format(set_type, f"{chunk_id:07}")
        print(
            "Downloading chunk {}/{}: {}".format(
                i + 1, len(chunk_ids_to_download), chunk_filename
            )
        )
        chunk_details = next(
            (item for item in metadata if item["filename"] == chunk_filename), None
        )
        if chunk_details is None:
            raise ValueError(f"Chunk metadata not found for {chunk_filename}")
        download_url = chunk_details["cdn"]
        print(download_url)
        download_sha = chunk_details["sha"]
        download_local_filename = os.path.join(output_dir, chunk_filename)
        with tqdm(
            unit="B", unit_scale=True, leave=True, miniters=1, desc="Progress"
        ) as t:  # all optional kwargs
            urllib.request.urlretrieve(
                download_url,
                download_local_filename,
                reporthook=urllib_tqdm_hook(t),
                data=None,
            )
            with open(str(download_local_filename), "rb") as f:
                local_sha = hashlib.sha1(f.read()).hexdigest()
            assert (
                local_sha == download_sha
            ), f"Downloaded file has a checksum that does not match the checksum in the metadata file. Expected: {download_sha}, actual: {local_sha}"

            if unzip_flag:
                extracted_count, found_files_by_scene = extract_required_files(
                    download_local_filename, output_dir, requested_scene_ids
                )
                warn_for_missing_files(
                    found_files_by_scene, requested_scene_ids, chunk_scene_ids
                )
                print(
                    f"Extracted {extracted_count} required files from {chunk_filename}"
                )
                os.remove(download_local_filename)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        cdn_file=args.cdn_file,
        output_dir=args.output_dir,
        scene_ids=args.scene_ids,
        set_type=args.set,
        unzip_flag=args.unzip,
    )
