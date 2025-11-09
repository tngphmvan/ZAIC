from PIL import Image
import numpy as np
from ultralytics import YOLO  # assuming you have ultralytics installed
import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import os
model = YOLOE("yoloe-11l-seg.pt")
# ... (your existing code for model and refer_model initialization) ...
refer_model = YOLO("yolo11l.pt")

home = "train/samples/Backpack_0"
object_images_dir = os.path.join(home, "object_images")
refer_images_filenames = os.listdir(object_images_dir)

# make sure we only pick actual image files
refer_images_paths = [os.path.join(object_images_dir, f) for f in refer_images_filenames if f.lower(
).endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

if len(refer_images_paths) < 3:
    print("not enough reference images to concatenate 3.")
else:
    # grab the first three images to concatenate
    images_to_concat = [Image.open(p).convert("RGB")
                        for p in refer_images_paths[:3]]

    # make sure all images have the same height for clean concatenation
    min_height = min(img.height for img in images_to_concat)
    resized_images = [img.resize(
        (int(img.width * min_height / img.height), min_height)) for img in images_to_concat]

    # calculate total width
    total_width = sum(img.width for img in resized_images)

    # create a new blank image for the concatenated result
    concatenated_image = Image.new('RGB', (total_width, min_height))

    # paste each image into the new canvas
    current_x_offset = 0
    for img in resized_images:
        concatenated_image.paste(img, (current_x_offset, 0))
        current_x_offset += img.width

    # you can save this if you want to see the result
    concatenated_image.save("concatenated_refer_images.jpg")
    print("concatenated_refer_images.jpg saved.")

# ... (after the concatenation code from step 1) ...

# convert the PIL image to a numpy array for ultralytics if it's not already handled
# ultralytics predict can often take PIL image directly, but numpy is a safe bet.
concatenated_np_image = np.array(concatenated_image)

# predict on the concatenated image
print("running prediction on concatenated image...")
concatenated_boxes_results = refer_model.predict(concatenated_np_image)

# extract bounding boxes from the results
if concatenated_boxes_results:
    # the result is usually a list of result objects, even for a single image
    first_result = concatenated_boxes_results[0]
    concatenated_bboxes = first_result.boxes.xyxy.cpu().numpy()
    print("bounding boxes from concatenated image:")
    print(concatenated_bboxes)
else:
    print("no objects detected in the concatenated image.")

visual_prompts = dict(
    bboxes=concatenated_bboxes,  # Box enclosing person
    cls=np.zeros(len(concatenated_bboxes)),  # ID to be assigned for person
)

# # Run prediction on a different image, using reference image to guide what to look for
results = model.track(
    "train/samples/Backpack_0/drone_video.mp4",  # Target image for detection
    refer_image=concatenated_np_image,  # Reference image used to get visual prompts
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    save=True,            # lưu kết quả
    save_txt=False,
    save_conf=True,
    stream=False,
    batch=64,
    half=True,
    verbose=False,
    show=True,
    # tracker_type="bytetrack"
)
