#!/usr/bin/env python
import os

__all__ = ["DataLoader"]


class DataLoader:
    def __init__(self, image_lists_path: str, cloud_lists_path: str, data_path: str):
        all_image_files = open(image_lists_path, "r").read().splitlines()
        all_cloud_files = open(cloud_lists_path, "r").read().splitlines()

        assert len(all_image_files) == len(all_cloud_files), "number of image and cloud files must be the same"
        assert len(all_image_files) > 0, "number of image files must be > 0"

        self.all_image_files = [os.path.join(data_path, image_file) for image_file in all_image_files]
        self.all_cloud_files = [os.path.join(data_path, cloud_file) for cloud_file in all_cloud_files]

        for _file in self.all_image_files:
            assert os.path.isfile(_file), f"{_file} does not exist"

        for _file in self.all_cloud_files:
            assert os.path.isfile(_file), f"{_file} does not exist"
