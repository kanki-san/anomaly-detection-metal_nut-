from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.checkpoint import checkpoint


def main():

    ROOT_PATH = Path("K:/Projects/fail2/metal_nut")

    # Folder datamodule
    datamodule = Folder(
        name="metal_nut_custom",
        root=ROOT_PATH,
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        num_workers=0,
    )


    datamodule.setup()

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )




    engine = Engine(
        accelerator="gpu",
        devices=1,
        max_epochs=10,
    )


    engine.train(datamodule=datamodule, model=model)
    engine.test(datamodule=datamodule, model=model)

if __name__ == '__main__':
    main()
