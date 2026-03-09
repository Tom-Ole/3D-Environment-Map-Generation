
import bosdyn.client
import bosdyn.client.util
from utils.get_images import get_image, GetImageOptions

def main(options):
    
    sdk = bosdyn.client.create_standard_sdk('image_depth_plus_visual')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    
    options: GetImageOptions = GetImageOptions()
    frames = get_image(robot, options)   
    
    return