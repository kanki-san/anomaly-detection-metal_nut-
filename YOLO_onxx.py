
from ultralytics import YOLO

if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()

    model = YOLO("yolov8m.pt")

    onxx_model = model.train(
        data=r"K:\Projects\fail2\yolo_metal_nut\dataset.yaml",
        epochs=200,
        imgsz=1024,
        batch=4,
        device=0,
        workers=4,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5,
        translate=0.1,
        scale=0.5,
        shear=2,

        flipud=0.0,
        fliplr=0.5,

        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        rect=False,
        patience=50,

        box=7.5,
        cls=0.5,

        project="onnx_metal",
        name="rtx5060",
        save=True)







