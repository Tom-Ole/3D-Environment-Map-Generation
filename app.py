import time
import bosdyn.client
import bosdyn.client.util
from utils.get_images import get_image, GetImageOptions

def main(options):
    
    sdk = bosdyn.client.create_standard_sdk('image_depth_plus_visual')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    
    options: GetImageOptions = GetImageOptions()

    rate = 10
    dt = 1.0 / rate

    while True:
        start = time.time()
        frame_id = 1

        get_image(robot, options, f"{frame_id:05d}")

        frame_id += 1
        elapsed = time.time() - start
        time.sleep(max(0, dt - elapsed))
    
    return
    
