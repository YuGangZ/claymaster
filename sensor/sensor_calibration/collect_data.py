import argparse
import os

import cv2
import numpy as np
import yaml

# from gs_sdk_master.gs_sdk.gs_device import Camera
from utils import load_csv_as_dict

"""
This script collects tactile data using ball indenters for sensor calibration.

Instruction: 
    1. Connect the sensor to the computer.
    2. Prepare a ball indenter with known diameter.
    3. Runs this script, press 'b' to collect a background image.
    4. Press the sensor with the ball indenter at multiple locations (~50 locations preferred),
    press 'w' to save the tactile image. When done, press 'q' to quit.
Note:
    If you have prepared multiple balls in different diameters, you can run this script multiple
    times, assign the same calib_dir but different ball diameters, the system will treat it as 
    one single dataset.

Usage:
    python collect_data.py --calib_dir CALIB_DIR --ball_diameter DIAMETER [--config_path CONFIG_PATH]

Arguments:
    --calib_dir: Path to the directory where the collected data will be saved
    --ball_diameter: Diameter of the ball indenter in mm
    --config_path: (Optional) Path to the configuration file about the sensor dimensions.
                If not provided, GelSight Mini is assumed.
"""

config_dir = os.path.join(os.path.dirname(__file__), "configs")
camera_matrix = np.array([[465.28398883, 0, 644.03369575], [0, 462.36047133, 475.80469358], [0, 0, 1]], dtype=np.float32)  # 相机矩阵
dist_coeffs = np.array([0.24424481, -0.28487431, -0.0005922, -0.00183754, 0.08188425], dtype=np.float32)  # 畸变系数
# 目标图像裁切区域 (y1:y2, x1:x2)
crop_x1, crop_x2 = 160, 1120  # 水平方向裁切
crop_y1, crop_y2 = 30, 930   # 垂直方向裁切
def collect_data():
    # Argument Parsers
    parser = argparse.ArgumentParser(
        description="Collect calibration data with ball indenters to calibrate the sensor."
    )
    parser.add_argument(
        "-b",
        "--calib_dir",
        type=str,
        help="path to save calibration data",
        default=os.path.join(os.path.dirname(__file__), "examples/calib_data_test"),
    )
    parser.add_argument(
        "-d", "--ball_diameter", type=float, help="diameter of the indenter in mm",
        default=10.0
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of the sensor information",
        default=os.path.join(config_dir, "mygelsight.yaml"),
    )
    args = parser.parse_args()

    # Create the data saving directories
    calib_dir = args.calib_dir
    ball_diameter = args.ball_diameter
    indenter_subdir = "%.3fmm" % (ball_diameter)
    indenter_dir = os.path.join(calib_dir, indenter_subdir)
    if not os.path.isdir(indenter_dir):
        os.makedirs(indenter_dir)

    # Read the configuration
    config_path = args.config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        imgh = config["raw_imgh"]
        imgw = config["raw_imgw"]
        save_imgh = config["imgh"]
        save_imgw = config["imgw"]

    # Create the data saving catalog
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    if not os.path.isfile(catalog_path):
        with open(catalog_path, "w") as f:
            f.write("experiment_reldir,diameter(mm)\n")

    # Find last data_count collected with this diameter
    data_dict = load_csv_as_dict(catalog_path)
    diameters = np.array([float(diameter) for diameter in data_dict["diameter(mm)"]])
    data_idxs = np.where(np.abs(diameters - ball_diameter) < 1e-3)[0]
    data_counts = np.array(
        [int(os.path.basename(reldir)) for reldir in data_dict["experiment_reldir"]]
    )
    if len(data_idxs) == 0:
        data_count = 0
    else:
        data_count = max(data_counts[data_idxs]) + 1

    # Connect to the device and collect data until quit
    cap = cv2.VideoCapture(1)

    # # 获取默认的曝光设置
    # default_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    # print(f"默认曝光值: {default_exposure}")
    # # 设置新的曝光值，范围通常是[-2, +2]，正值增大曝光，负值减小曝光
    # new_exposure = default_exposure
    # # 设置新的曝光
    # cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
    # cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgh)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Press key to collect data, collect background, or quit (w/b/q)")
    while True:
        _, image = cap.read()

        # 进行去畸变
        h, w = image.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_K)

        # 进行裁切
        image = undistorted_image[crop_y1:crop_y2, crop_x1:crop_x2]
        image = cv2.resize(image, (save_imgw, save_imgh), interpolation=cv2.INTER_AREA)

        # Display the image and decide record or quit
        cv2.imshow("frame", image)
        key = cv2.waitKey(100)
        if key == ord("w"):
            # Save the image
            experiment_reldir = os.path.join(indenter_subdir, str(data_count))
            experiment_dir = os.path.join(calib_dir, experiment_reldir)
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)
            save_path = os.path.join(experiment_dir, "gelsight.png")
            cv2.imwrite(save_path, image)
            print("Save data to new path: %s" % save_path)

            # Save to catalog
            with open(catalog_path, "a") as f:
                f.write(experiment_reldir + "," + str(ball_diameter))
                f.write("\n")
            data_count += 1
        elif key == ord("b"):
            print("Collecting 10 background images, please wait ...")
            images = []
            for _ in range(10):
                _, image = cap.read()
                h, w = image.shape[:2]
                new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
                undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_K)
                cropped_image = undistorted_image[crop_y1:crop_y2, crop_x1:crop_x2]
                image = cv2.resize(cropped_image, (save_imgw, save_imgh), interpolation=cv2.INTER_AREA)
                images.append(image)
                cv2.imshow("frame", image)
                cv2.waitKey(1)
            image = np.mean(images, axis=0).astype(np.uint8)
            # Save the background image
            save_path = os.path.join(calib_dir, "background.png")
            cv2.imwrite(save_path, image)
            print("Save background image to %s" % save_path)
        elif key == ord("q"):
            # Quit
            break
        elif key == -1:
            # No key pressed
            continue
        else:
            print("Unrecognized key %s" % key)

    cap.release()
    cv2.destroyAllWindows()
    print("%d images collected in total." % data_count)


if __name__ == "__main__":
    collect_data()
