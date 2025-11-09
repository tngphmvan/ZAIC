from __future__ import annotations

from sahi.models.ultralytics import UltralyticsDetectionModel
import numpy as np

class YOLOEDetectionModel(UltralyticsDetectionModel):
    """YOLOE Detection Model for open-vocabulary detection and segmentation.

    YOLOE (Real-Time Seeing Anything) is a zero-shot, promptable YOLO model designed for
    open-vocabulary detection and segmentation. It supports text prompts, visual prompts,
    and prompt-free detection with internal vocabulary (1200+ categories).

    Key Features:
        - Open-vocabulary detection: Detect any object class via text prompts
        - Visual prompting: One-shot detection using reference images
        - Instance segmentation: Built-in segmentation for detected objects
        - Real-time performance: Maintains YOLO speed with no inference overhead
        - Prompt-free mode: Uses internal vocabulary for open-set recognition

    Available Models:
        Text/Visual Prompt models:
            - yoloe-11s-seg.pt, yoloe-11m-seg.pt, yoloe-11l-seg.pt
            - yoloe-v8s-seg.pt, yoloe-v8m-seg.pt, yoloe-v8l-seg.pt

        Prompt-free models:
            - yoloe-11s-seg-pf.pt, yoloe-11m-seg-pf.pt, yoloe-11l-seg-pf.pt
            - yoloe-v8s-seg-pf.pt, yoloe-v8m-seg-pf.pt, yoloe-v8l-seg-pf.pt

    !!! example "Usage with text prompts"
        ```python
        from sahi import AutoDetectionModel

        # Load YOLOE model
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yoloe",
            model_path="yoloe-11l-seg.pt",
            confidence_threshold=0.3,
            device="cuda:0"
        )

        # Set text prompts for specific classes
        detection_model.model.set_classes(
            ["person", "car", "traffic light"],
            detection_model.model.get_text_pe(["person", "car", "traffic light"])
        )

        # Perform prediction
        from sahi.predict import get_prediction
        result = get_prediction("image.jpg", detection_model)
        ```

    !!! example "Usage for standard detection (no prompts)"
        ```python
        from sahi import AutoDetectionModel

        # Load YOLOE model (works like standard YOLO)
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yoloe",
            model_path="yoloe-11l-seg.pt",
            confidence_threshold=0.3,
            device="cuda:0"
        )

        # Perform prediction without prompts (uses internal vocabulary)
        from sahi.predict import get_sliced_prediction
        result = get_sliced_prediction(
            "image.jpg",
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        ```

    Note:
        - YOLOE models perform instance segmentation by default
        - When used without prompts, YOLOE performs like standard YOLO11 with identical speed
        - For visual prompting, see Ultralytics YOLOE documentation
        - YOLOE achieves +3.5 AP over YOLO-Worldv2 on LVIS with 1.4x faster inference

    References:
        - Paper: https://arxiv.org/abs/2503.07465
        - Docs: https://docs.ultralytics.com/models/yoloe/
        - GitHub: https://github.com/THU-MIG/yoloe
    """
    def __init__(self, refer_image,  visual_prompts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refer_image = refer_image
        self.visual_prompts = visual_prompts
    def load_model(self):
        """Loads the YOLOE detection model from the specified path.

        Initializes the YOLOE model with the given model path or uses the default
        'yoloe-11s-seg.pt' if no path is provided. The model is then moved to the
        specified device (CPU/GPU).

        By default, YOLOE works in prompt-free mode using its internal vocabulary
        of 1200+ categories. To use text prompts for specific classes, call
        model.set_classes() after loading:

            model.set_classes(["person", "car"], model.get_text_pe(["person", "car"]))

        Raises:
            TypeError: If the model_path is not a valid YOLOE model path or if
                      the ultralytics package with YOLOE support is not installed.
        """
        from ultralytics import YOLOE

        try:
            model_source = self.model_path or "yoloe-11s-seg.pt"
            model = YOLOE(model_source)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError(f"model_path is not a valid YOLOE model path: {e}") from e
    
    
    def perform_inference(self, image: np.ndarray):
        """Prediction is performed using self.model and the prediction result is set to self._original_predictions.

        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded

        import torch

        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        kwargs = {"cfg": self.config_path, "verbose": False, "conf": self.confidence_threshold, "device": self.device, "refer_image": self.refer_image, "save":True,            # lưu kết quả
    "save_txt":False,
    "save_conf":True,
    "stream":False,
    "batch":64,
    "half":True,
    "verbose":False}

        if self.image_size is not None:
            kwargs = {"imgsz": self.image_size, **kwargs}

        prediction_result = self.model(image[:, :, ::-1], **kwargs)  # YOLO expects numpy arrays to have BGR

        # Handle different result types for PyTorch vs ONNX models
        # ONNX models might return results in a different format
        if self.has_mask:
            from ultralytics.engine.results import Masks

            if not prediction_result[0].masks:
                # Create empty masks if none exist
                if hasattr(self.model, "device"):
                    device = self.model.device
                else:
                    device = "cpu"  # Default for ONNX models
                prediction_result[0].masks = Masks(
                    torch.tensor([], device=device), prediction_result[0].boxes.orig_shape
                )

            # We do not filter results again as confidence threshold is already applied above
            prediction_result = [
                (
                    result.boxes.data,
                    result.masks.data,
                )
                for result in prediction_result
            ]
        elif self.is_obb:
            # For OBB task, get OBB points in xyxyxyxy format
            device = getattr(self.model, "device", "cpu")
            prediction_result = [
                (
                    # Get OBB data: xyxy, conf, cls
                    torch.cat(
                        [
                            result.obb.xyxy,  # box coordinates
                            result.obb.conf.unsqueeze(-1),  # confidence scores
                            result.obb.cls.unsqueeze(-1),  # class ids
                        ],
                        dim=1,
                    )
                    if result.obb is not None
                    else torch.empty((0, 6), device=device),
                    # Get OBB points in (N, 4, 2) format
                    result.obb.xyxyxyxy if result.obb is not None else torch.empty((0, 4, 2), device=device),
                )
                for result in prediction_result
            ]
        else:  # If model doesn't do segmentation or OBB then no need to check masks
            # We do not filter results again as confidence threshold is already applied above
            prediction_result = [result.boxes.data for result in prediction_result]

        self._original_predictions = prediction_result
        self._original_shape = image.shape