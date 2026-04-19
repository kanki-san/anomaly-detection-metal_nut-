from pathlib import Path
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO
from anomalib.models import Patchcore

 #load previsios create model or tensor
# this funtion was created so all data of patchcore may be loaded
# patchcore require model structure, parameter, not only weights
# *args are positional arguments(values store in tuple
#**kwargs are key_values pairs eg dict as weighrt are stored as kwargs
_original_torch_load = torch.load


def fixed_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = fixed_torch_load



@dataclass
class CombinedResult:
    image_path: Path
    yolo_detections: List[Dict]
    patchcore_score: float
    patchcore_anomaly_map: np.ndarray
    is_anomaly: bool
    combined_risk_score: float
    visualization: np.ndarray



class CombinedAnomalyDetector:
    def __init__(
            self,
            patchcore_checkpoint: Path,
            yolo_model_path: Path,
            patchcore_threshold: float = 0.5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.patchcore_threshold = patchcore_threshold

        # Load PatchCore
        self.patchcore_model = Patchcore.load_from_checkpoint(
            patchcore_checkpoint,
            map_location=self.device
        )
        self.patchcore_model.eval().freeze().to(device)


        self.patchcore_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])


        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(device)


    def _run_patchcore(self, image: Image.Image) -> Tuple[float, np.ndarray]:
        input_tensor = self.patchcore_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.patchcore_model(input_tensor)

        # Handle dict output (standard anomalib)
        if isinstance(predictions, dict):
            score = float(predictions["pred_scores"].cpu().numpy().flatten()[0])
            # Try to get anomaly map, fallback to dummy if not available
            if "anomaly_maps" in predictions:
                anomaly_map = predictions["anomaly_maps"].cpu().numpy()
            else:
                # Create dummy 2D map from score
                anomaly_map = np.full((64, 64), score, dtype=np.float32)

        # Handle tuple/list output
        elif isinstance(predictions, (list, tuple)):
            score = float(predictions[0].cpu().numpy().flatten()[0])
            if len(predictions) > 1 and predictions[1] is not None:
                anomaly_map = predictions[1].cpu().numpy()
            else:
                anomaly_map = np.full((64, 64), score, dtype=np.float32)


        else:
            pred_np = predictions.cpu().numpy()
            score = float(pred_np.flatten()[0])

            anomaly_map = np.full((64, 64), score, dtype=np.float32)


        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.cpu().numpy()


        while anomaly_map.ndim > 2:
            anomaly_map = anomaly_map.squeeze(0)


        if anomaly_map.ndim == 1:
            if anomaly_map.size == 1:

                val = float(anomaly_map[0])
                anomaly_map = np.full((64, 64), val, dtype=np.float32)
            else:

                size = int(np.sqrt(anomaly_map.size))
                anomaly_map = anomaly_map[:size * size].reshape(size, size)


        anomaly_map = anomaly_map.astype(np.float32)

        return score, anomaly_map


    def _run_yolo(self, image: np.ndarray) -> List[Dict]:
        results = self.yolo_model(image, conf=0.25, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())

                    if conf < 0.5:
                        continue

                    detections.append({
                        "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                        "confidence": conf,
                        "class_name": result.names[int(box.cls[0])]
                    })

        return detections


    def _combine_scores(self, patchcore_score: float, yolo_detections: List[Dict]) -> float:
        """Combine PatchCore and YOLO scores."""
        if not yolo_detections:
            return patchcore_score


        max_yolo_conf = max(d["confidence"] for d in yolo_detections)
        # Weighted: 70% PatchCore, 30% YOLO influence
        combined = 0.7 * patchcore_score + 0.3 * max_yolo_conf
        return min(combined, 1.0)


    def _create_visualization(
            self,
            image: np.ndarray,
            anomaly_map: np.ndarray,
            detections: List[Dict],
            patchcore_score: float,
            combined_score: float,
            is_anomaly: bool
    ) -> np.ndarray:

        h, w = image.shape[:2]


        anomaly_map = np.asarray(anomaly_map, dtype=np.float32)


        anomaly_resized = cv2.resize(anomaly_map, (w, h))


        amin, amax = anomaly_resized.min(), anomaly_resized.max()
        if amax > amin:
            anomaly_norm = (anomaly_resized - amin) / (amax - amin)
        else:
            anomaly_norm = np.zeros_like(anomaly_resized)

        anomaly_uint8 = (anomaly_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(anomaly_uint8, cv2.COLORMAP_JET)


        if is_anomaly:
            overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        else:
            overlay = image.copy()


        if is_anomaly and detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                cls_name = det["class_name"]

                color = (0, 0, 255)

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    overlay,
                    f"{cls_name}: {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )


        status_text = "ANOMALY" if is_anomaly else "NORMAL"
        status_color = (0, 0, 255) if is_anomaly else (0, 255, 0)


        lines = [
            f"{status_text}",
            f"PatchCore: {patchcore_score:.3f}",
            f"Combined: {combined_score:.3f}",
            f"Detections: {len(detections)}"
        ]

        y_offset = 30
        for line in lines:
            cv2.putText(
                overlay,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )
            y_offset += 35

        return overlay


    def predict_single(self, image_path: Path) -> CombinedResult:
        image_pil = Image.open(image_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


        patchcore_score, anomaly_map = self._run_patchcore(image_pil)


        yolo_detections = self._run_yolo(image_cv)


        combined_score = self._combine_scores(patchcore_score, yolo_detections)
        is_anomaly = combined_score > self.patchcore_threshold


        vis = self._create_visualization(
            image_cv,
            anomaly_map,
            yolo_detections,
            patchcore_score,
            combined_score,
            is_anomaly
        )

        return CombinedResult(
            image_path=image_path,
            yolo_detections=yolo_detections,
            patchcore_score=patchcore_score,
            patchcore_anomaly_map=anomaly_map,
            is_anomaly=is_anomaly,
            combined_risk_score=combined_score,
            visualization=vis
        )

    def predict_folder(
            self,
            folder_path: Path,
            output_dir: Path = None,
            save_vis: bool = True
    ) -> List[CombinedResult]:
        """Predict folder of images."""
        folder_path = Path(folder_path)
        image_paths = [
            p for p in folder_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        ]

        results = []
        for img_path in image_paths:
            print(f"Processing: {img_path.name}")
            result = self.predict_single(img_path)
            results.append(result)

            print(f"  -> Anomaly: {result.is_anomaly}, "
                  f"Score: {result.patchcore_score:.3f}, "
                  f"Combined: {result.combined_risk_score:.3f}, "
                  f"YOLO dets: {len(result.yolo_detections)}")

            if save_vis and output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"result_{img_path.stem}.jpg"
                cv2.imwrite(str(out_path), result.visualization)

        return results



if __name__ == "__main__":
    detector = CombinedAnomalyDetector(
        patchcore_checkpoint=Path(
            r"K:\Projects\fail2\results\Patchcore\metal_nut_custom\v0\weights\lightning\model.ckpt"),
        yolo_model_path=Path(r"K:\Projects\fail2\runs\detect\onnx_metal\rtx5060\weights\best.pt"),
        patchcore_threshold=0.5
    )


    folder_path = Path(r"C:\Users\OUKI\Desktop\image_test")

    results = detector.predict_folder(
        folder_path=folder_path,
        output_dir=Path(r"C:\Users\OUKI\Desktop\output_results"),
        save_vis=True
    )

    print(f"\nProcessed {len(results)} images")