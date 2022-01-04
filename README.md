# 📝 lidar camera calibration #
***

<p align="center">
  <img src="./docs/images/result.gif">
</p>


## :tada: TODO ##
***

- [x] extrinsic calibration between a monocular camera and a 3D Lidar by using planar point to plane constraint
- [x] test the algorithm on real dataset. This is the public dataset from [the following repository of SubMishMar](https://github.com/SubMishMar/cam_lidar_calib)

## 🎛  Dependencies ##
***

```bash
conda env create --file environment.yml
conda activate lidar_camera_calibration
```

## :running: How to Run ##
***
- Download dataset from [HERE](https://drive.google.com/file/d/1VaDvPGEmekPpPBh9-JnUTDF8UVH1mXMl/view?usp=sharing) and extract into [./data repository](./data)
```bash
tar -xf ./data/dataset.tar.xz -C ./data/
```

- Run calibration app after building the required environment

```bash
python scripts/calibration_app.py --dataset_info_json ./data/dataset/dataset_info.json \
                                  --image_list_path ./data/dataset/image_list.txt \
                                  --cloud_list_path ./data/dataset/cloud_list.txt \
                                  --data_path ./data/dataset/data/
```

you can check the result of lidar point projection on images at /tmp/projected_image_*.jpg

**Disclaimer: the result does not look so great on this test, but I hope this repository can still be a template for a standard camera-lidar calibration project**

## :gem: References ##
***

- [Extrinsic Calibration of a 3D Laser Scanner and an Omnidirectional Camera, by Pandey et al.](https://www.sciencedirect.com/science/article/pii/S1474667016350790)
- [Extrinsic Calibration of a Camera and 2d Laser by MegviiRobot](https://github.com/MegviiRobot/CamLaserCalibraTool)
