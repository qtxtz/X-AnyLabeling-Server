import numpy as np
from loguru import logger
from PIL import Image
from typing import Any, Dict

from . import BaseModel
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model("pp_doclayout_v3")
class PPDocLayoutV3(BaseModel):
    """PP-DocLayoutV3 document layout detection model."""

    def load(self):
        """Load PP-DocLayoutV3 model and initialize components."""
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
        )

        model_path = self.params.get(
            "model_path", "PaddlePaddle/PP-DocLayoutV3_safetensors"
        )
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.device = device
        self.model = AutoModelForObjectDetection.from_pretrained(model_path)
        self.model.to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_path)

        logger.info(f"PP-DocLayoutV3 model loaded on {device}")

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute layout detection on input image.

        Args:
            image: Input image in BGR format.
            params: Inference parameters including confidence threshold.

        Returns:
            Dictionary with detected layout shapes.
        """
        import torch

        conf_threshold = params.get("conf_threshold", 0.5)
        pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[pil_image.size[::-1]]
        )

        shapes = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]
            polygon_points_list = result.get("polygon_points", [])

            has_polygons = len(polygon_points_list) == len(scores)

            for idx, (score, label_id, box) in enumerate(
                zip(scores, labels, boxes)
            ):
                score_val = score.item()
                if score_val < conf_threshold:
                    continue

                label = self.model.config.id2label[label_id.item()]

                if has_polygons and idx < len(polygon_points_list):
                    poly_pts = polygon_points_list[idx]
                    pts = (
                        poly_pts.tolist()
                        if hasattr(poly_pts, "tolist")
                        else list(poly_pts)
                    )
                    points = [[float(p[0]), float(p[1])] for p in pts]
                else:
                    box_list = box.tolist()
                    xmin, ymin, xmax, ymax = box_list
                    points = [
                        [float(xmin), float(ymin)],
                        [float(xmax), float(ymin)],
                        [float(xmax), float(ymax)],
                        [float(xmin), float(ymax)],
                    ]

                shape = Shape(
                    label=label,
                    shape_type="polygon",
                    points=points,
                    score=float(score_val),
                    group_id=idx + 1,
                )
                shapes.append(shape)

        return {"shapes": shapes}

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
