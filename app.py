import time
import bosdyn.client
import bosdyn.client.util
from utils.get_images import get_image, GetImageOptions
import argparse
import keyboard

def main(args):
    
    sdk = bosdyn.client.create_standard_sdk("image_depth_plus_visual")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    
    image_options: GetImageOptions = GetImageOptions()

    rate = 10
    dt = 1.0 / rate

    image_results = []

    while True:
        start = time.time()
        frame_id = 1

        get_image(robot, image_options, f"{frame_id:05d}", image_results)

        frame_id += 1
        elapsed = time.time() - start
        time.sleep(max(0, dt - elapsed))

        if(len(image_results) >= 100 or keyboard.is_pressed("q")):
            break
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Capture images from Spot's cameras.")
    parser.add_argument("--hostname", help="Hostname or address of robot,"
                        " e.g. 'beta25-p' or '192.168.80.3'", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print debug-level messages")
    args = parser.parse_args()

    main(args)
