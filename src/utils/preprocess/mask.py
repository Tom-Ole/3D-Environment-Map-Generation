import logging

from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
import torch

#yolo26n-seg.pt

logger = logging.getLogger(__name__)

class Preprocessor:

    model = YOLO("yolo26n-seg.pt")

    def __init__(self):
        pass

    def get_classes(self):
        """Return a dict of classes. For example: {0: 'person', 1: 'bicycle', 2: 'car', ..., 56: 'chair', ...}"""
        return self.model.names

    def classes_to_ids(self, classes: list[str]) -> list[int]:
        """Convert a list of class names to a list of class ids. For example: ['person', 'bicycle', 'car'] -> [0, 1, 2]"""
        class_ids = []
        for cls in classes:
            for id, name in self.model.names.items():
                if name == cls:
                    class_ids.append(id)
                    break
        return class_ids

    def create_masks(self, input_path: str, output_path: str, classes: list[int] = [0]):
        try:
            os.makedirs(output_path, exist_ok=True)

            all_files_name = os.listdir(input_path)
            all_files = [os.path.join(input_path, f) for f in all_files_name if f.endswith(('.jpg', '.png'))]

            if not all_files:
                logger.warning(f"No image files found in {input_path}")
                return

            results = self.model(all_files, classes=classes)

            c = 0
            for result in results:
                try:
                    base_name = os.path.splitext(all_files_name[c])[0]
                    file_name_path = os.path.join(output_path, f"{base_name}_mask.png")

                    masks = result.masks

                    if masks is not None:
                        combined = masks.data.any(dim=0).cpu().numpy()

                        orig_h, orig_w = result.orig_shape
                        mask_img = Image.fromarray((combined * 255).astype(np.uint8), mode="L")
                        mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

                        mask_img.save(file_name_path)
                        logger.info(f"Saved mask for {all_files_name[c]} at {file_name_path}")
                    else:
                        orig_h, orig_w = result.orig_shape
                        blank = Image.fromarray(np.zeros((orig_h, orig_w), dtype=np.uint8), mode="L")
                        blank.save(file_name_path)
                        logger.info(f"No masks found for {all_files_name[c]}. Saved blank mask at {file_name_path}")

                    c += 1
                except Exception as e:
                    logger.error(f"Failed to process {all_files_name[c]}: {e}")
                    c += 1
                    continue
        except Exception as e:
            logger.error(f"Failed to create masks: {e}")
            raise


    def create_masks_recursive(self, input_path: str, output_path: str, classes: list[int] = [0]):
        try:
            output_path = output_path + "/masks"
            
            for root, dirs, files in os.walk(input_path):
                image_files = [f for f in files if f.endswith(('.jpg', '.png'))]
                if not image_files:
                    continue

                try:
                    relative = os.path.relpath(root, input_path)
                    current_output_path = os.path.join(output_path, relative)
                    os.makedirs(current_output_path, exist_ok=True)

                    all_files = [os.path.join(root, f) for f in image_files]
                    results = self.model(all_files, classes=classes)

                    for i, result in enumerate(results):
                        try:
                            base_name = os.path.splitext(image_files[i])[0]
                            file_name_path = os.path.join(current_output_path, f"{base_name}_mask.png")

                            masks = result.masks
                            orig_h, orig_w = result.orig_shape

                            if masks is not None:
                                combined = masks.data.any(dim=0).cpu().numpy()
                                mask_img = Image.fromarray((combined * 255).astype(np.uint8), mode="L")
                                mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
                                mask_img.save(file_name_path)
                                logger.info(f"Saved mask for {image_files[i]} at {file_name_path}")
                            else:
                                blank = Image.fromarray(np.zeros((orig_h, orig_w), dtype=np.uint8), mode="L")
                                blank.save(file_name_path)
                                logger.info(f"No masks found for {image_files[i]}. Saved blank mask at {file_name_path}")
                        except Exception as e:
                            logger.error(f"Failed to process {image_files[i]}: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Failed to process directory {root}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Failed to create masks recursively: {e}")
            raise



if __name__ == "__main__":
    preprocessor = Preprocessor()
    #print(preprocessor.get_classes())
    preprocessor.create_masks(input_path="./output/20260520_14_3629/images/images/back_fisheye_image/", output_path="./output/20260520_14_3629/masks")

    