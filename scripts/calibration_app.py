#!/usr/bin/env python
import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
if True:
    from lidar_camera_calibration import CalibrationHandler


def get_args():
    parser = argparse.ArgumentParser("lrf and camera calibration")
    parser.add_argument("--dataset_info_json", type=str, required=True)
    parser.add_argument("--image_lists_path", type=str, required=True)
    parser.add_argument("--cloud_lists_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    calibration_handler = CalibrationHandler(
        args.dataset_info_json, args.image_lists_path, args.cloud_lists_path, args.data_path
    )
    calibration_handler.run()


if __name__ == "__main__":
    main()
