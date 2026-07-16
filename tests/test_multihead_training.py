"""Tests for training the multi-head (explainability) CNN through the Lightning module.

Covers the regression introduced by the lazy `load_model` refactor: `ClassificationLitModule.setup()`
calls `net.load_model()`, which only exists on `BaseModel`, so `train_cnn_multi` crashes at the
start of `trainer.fit()`. Also covers handling of the `(output, cam)` tuple returned when
`multi_head` is enabled and the ONNX export of a two-output model.
"""

import functools

import pytest
import torch
from torchvision import transforms as T

from src.data.classification_datamodule import ClassificationDataModule
from src.models.classification_module import ClassificationLitModule
from src.models.components.cnn_cam_multihead import CNNCAMMultihead
from src.models.components.utils import export_model_to_onnx
from src.models.components.vit_rollout_multihead import VitRolloutMultihead
from tests.helpers.digits_dataset import generate_classification_dataset
from tests.helpers.experiment_runner import assert_exported_classifier


def make_cnn_cam_net(multi_head: bool) -> CNNCAMMultihead:
    return CNNCAMMultihead(
        backbone='torchvision.models/mobilenet_v3_large',
        multi_head=multi_head,
        return_node='features.13.block.0',
        weights=None,
    )


def make_vit_rollout_net(multi_head: bool, output_size: int = 1) -> VitRolloutMultihead:
    return VitRolloutMultihead(
        backbone='timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k',
        multi_head=multi_head,
        pretrained=False,
        output_size=output_size,
        return_nodes='attn_drop',
        head_name='head',
        img_size=224,
        discard_ratio=0.2,
        head_fusion='mean',
    )


def make_module(
    net: torch.nn.Module,
    loss: torch.nn.Module = None,
    num_classes: int = 2,
) -> ClassificationLitModule:
    return ClassificationLitModule(
        net=net,
        optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
        scheduler=None,
        loss=loss if loss is not None else torch.nn.BCELoss(),
        compile=False,
        ckpt_path=None,
        num_classes=num_classes,
    )


def test_setup_with_cnn_cam_multihead():
    module = make_module(make_cnn_cam_net(multi_head=True))
    module.setup(stage='fit')


def test_cnn_cam_multihead_load_model_rejects_multiclass():
    net = make_cnn_cam_net(multi_head=True)
    with pytest.raises(ValueError):
        net.load_model(num_classes=5)


def test_model_step_with_multi_head_output():
    module = make_module(make_cnn_cam_net(multi_head=True))
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2,))

    loss, preds, targets = module.model_step((x, y))

    assert loss.ndim == 0
    assert preds.shape == (2,)
    assert targets.shape == (2,)


class _FixedProbabilityNet(torch.nn.Module):
    """Emits fixed probabilities, mimicking CNNCAMMultihead's sigmoid output."""

    def __init__(self, probs: torch.Tensor):
        super().__init__()
        self.probs = probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probs


def test_model_step_thresholds_probabilities_at_half():
    module = make_module(_FixedProbabilityNet(torch.tensor([[0.3], [0.9]])))
    x = torch.zeros(2, 3, 8, 8)
    y = torch.tensor([0, 1])

    _, preds, _ = module.model_step((x, y))

    assert preds.tolist() == [0.0, 1.0]


def test_predict_step_with_multi_head_output():
    module = make_module(make_cnn_cam_net(multi_head=True))
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2,))

    preds = module.predict_step((x, y), batch_idx=0)

    assert preds.shape == (2,)


def test_predict_step_thresholds_probabilities_at_half():
    module = make_module(_FixedProbabilityNet(torch.tensor([[0.3], [0.9]])))
    x = torch.zeros(2, 3, 8, 8)
    y = torch.tensor([0, 1])

    preds = module.predict_step((x, y), batch_idx=0)

    assert preds.tolist() == [0.0, 1.0]


def test_setup_with_vit_rollout_multihead():
    module = make_module(make_vit_rollout_net(multi_head=True))
    module.setup(stage='fit')


def test_vit_load_model_rejects_mismatched_multiclass():
    net = make_vit_rollout_net(multi_head=True, output_size=1)
    with pytest.raises(ValueError):
        net.load_model(num_classes=5)


def test_vit_load_model_accepts_matching_multiclass():
    net = make_vit_rollout_net(multi_head=True, output_size=5)
    net.load_model(num_classes=5)


def test_model_step_with_vit_multi_head_output():
    module = make_module(make_vit_rollout_net(multi_head=True))
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2,))

    loss, preds, targets = module.model_step((x, y))

    assert loss.ndim == 0
    assert preds.shape == (2,)
    assert targets.shape == (2,)


def test_model_step_multiclass_with_vit_multi_head():
    module = make_module(
        make_vit_rollout_net(multi_head=True, output_size=3),
        loss=torch.nn.CrossEntropyLoss(),
        num_classes=3,
    )
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0, 2])

    loss, preds, targets = module.model_step((x, y))

    assert loss.ndim == 0
    assert preds.shape == (2,)
    assert set(preds.tolist()) <= {0, 1, 2}


def test_vit_multi_head_onnx_export_loads_and_infers(tmp_path):
    net = make_vit_rollout_net(multi_head=True)
    onnx_path = str(tmp_path / 'vit_model.onnx')

    export_model_to_onnx(
        net,
        onnx_path,
        input_shape=(1, 3, 224, 224),
    )

    assert_exported_classifier(onnx_path, class_neurons=1, expect_map=True)


def test_vit_multiclass_multi_head_onnx_export_loads_and_infers(tmp_path):
    net = make_vit_rollout_net(multi_head=True, output_size=3)
    onnx_path = str(tmp_path / 'vit_model_mc.onnx')

    export_model_to_onnx(
        net,
        onnx_path,
        input_shape=(1, 3, 224, 224),
    )

    assert_exported_classifier(onnx_path, class_neurons=3, expect_map=True)


def test_multi_head_onnx_export_loads_and_infers(tmp_path):
    net = make_cnn_cam_net(multi_head=True)
    onnx_path = str(tmp_path / 'model.onnx')

    # same call shape as src/train.py (default output_names)
    export_model_to_onnx(
        net,
        onnx_path,
        input_shape=(1, 3, 224, 224),
    )

    assert_exported_classifier(onnx_path, class_neurons=1, expect_map=True)


@pytest.mark.parametrize('source_channels', [1, 3])
def test_pipeline_handles_grey_and_rgb_source_images(tmp_path, source_channels):
    """ImageFolder must deliver 3-channel batches to the model regardless of
    whether the source images on disk are grayscale or RGB."""
    splits = generate_classification_dataset(
        tmp_path / 'dataset', num_classes=2, channels=source_channels
    )
    transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    datamodule = ClassificationDataModule(
        train_data_dir=splits['train'],
        test_data_dir=splits['test'],
        val_data_dir=splits['val'],
        batch_size=4,
        num_workers=1,
        train_transforms=transforms,
        val_test_transforms=transforms,
    )
    datamodule.setup(stage='fit')
    x, y = next(iter(datamodule.train_dataloader()))
    assert x.shape == (4, 3, 224, 224)

    module = make_module(make_cnn_cam_net(multi_head=True))
    loss, preds, targets = module.model_step((x, y))
    assert loss.ndim == 0
    assert preds.shape == (4,)
