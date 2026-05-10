from enum import Enum

class ImageSources(Enum):
    BACK_DEPTH_IN_VISUAL_FRAME = "back_depth_in_visual_frame"
    BACK_DEPTH = "back_depth"
    BACK_FISHEYE_IMAGE = "back_fisheye_image"
    FRONTLEFT_DEPTH = "frontleft_depth"
    FRONTLEFT_DEPTH_IN_VISUAL_FRAME = "frontleft_depth_in_visual_frame"
    FRONTLEFT_FISHEYE_IMAGE = "frontleft_fisheye_image"
    FRONTRIGHT_DEPTH = "frontright_depth"
    FRONTRIGHT_DEPTH_IN_VISUAL_FRAME = "frontright_depth_in_visual_frame"
    FRONTRIGHT_FISHEYE_IMAGE = "frontright_fisheye_image"
    LEFT_DEPTH = "left_depth"
    LEFT_DEPTH_IN_VISUAL_FRAME = "left_depth_in_visual_frame"
    LEFT_FISHEYE_IMAGE = "left_fisheye_image"
    RIGHT_DEPTH = "right_depth"
    RIGHT_DEPTH_IN_VISUAL_FRAME = "right_depth_in_visual_frame"
    RIGHT_FISHEYE_IMAGE = "right_fisheye_image"

    def get_color(self):
        return [
            self.BACK_FISHEYE_IMAGE,
            self.FRONTLEFT_FISHEYE_IMAGE,
            self.FRONTRIGHT_FISHEYE_IMAGE,
            self.LEFT_FISHEYE_IMAGE,
            self.RIGHT_FISHEYE_IMAGE,
        ]

    def get_depth(self):
        return [
            self.BACK_DEPTH,
            self.FRONTLEFT_DEPTH,
            self.FRONTRIGHT_DEPTH,
            self.LEFT_DEPTH,
            self.RIGHT_DEPTH,
        ]
    
    def get_depth_in_visual_frame(self):
        return [
            self.BACK_DEPTH_IN_VISUAL_FRAME,
            self.FRONTLEFT_DEPTH_IN_VISUAL_FRAME,
            self.FRONTRIGHT_DEPTH_IN_VISUAL_FRAME,
            self.LEFT_DEPTH_IN_VISUAL_FRAME,
            self.RIGHT_DEPTH_IN_VISUAL_FRAME,
        ]
    
    def __str__(self):
        return super().__str__()


class ImageOptions:
    output_path: str
    sources: list[ImageSources] | None = None

    correct_rotation: bool = True
    show: bool = False
    save: bool = True

    side_tilt: bool = False
    side_tile_angle: float = 15.0
    tilt_settle_time: float = 1.0

    def __init__(self, output_path: str, sources: list[ImageSources] | None = None, ):
        self.output_path = output_path
        self.sources = sources

    def list_sources(self, print: bool = True) -> list[str]:
        print(ImageSources)
        return [source.value for source in ImageSources]
    