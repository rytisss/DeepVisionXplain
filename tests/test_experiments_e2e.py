"""End-to-end train() runs for every classification experiment config.

Each test runs the real training task (fit, test, ONNX export) on a small
dataset generated from the scikit-learn digits data, then loads the exported
model with onnxruntime and runs inference. Together they cover both
explainability (multi-head) and plain experiments, binary and multiclass
heads, and grayscale and RGB source images.
"""

import pytest

from tests.helpers.experiment_runner import (
    assert_exported_classifier,
    run_eval_on_digits_dataset,
    run_experiment_on_digits_dataset,
)


@pytest.mark.slow
def test_train_cnn_multi_end_to_end(tmp_path):
    # grayscale source images; the other tests cover the RGB source path
    onnx_path = run_experiment_on_digits_dataset(
        tmp_path,
        experiment='train_cnn_multi',
        num_classes=2,
        extra_overrides=['model.net.weights=null'],
        source_channels=1,
    )
    assert_exported_classifier(onnx_path, class_neurons=1, expect_map=True)


@pytest.mark.slow
def test_train_vit_multi_multiclass_end_to_end(tmp_path):
    onnx_path = run_experiment_on_digits_dataset(
        tmp_path,
        experiment='train_vit_multi',
        num_classes=3,
        extra_overrides=[
            'model.net.pretrained=false',
            'model.net.output_size=3',
            'model.loss._target_=torch.nn.CrossEntropyLoss',
        ],
    )
    assert_exported_classifier(onnx_path, class_neurons=3, expect_map=True)


@pytest.mark.slow
def test_train_cnn_end_to_end(tmp_path):
    onnx_path = run_experiment_on_digits_dataset(
        tmp_path,
        experiment='train_cnn',
        num_classes=2,
        extra_overrides=['model.net.pretrained=false'],
    )
    assert_exported_classifier(onnx_path, class_neurons=2)


@pytest.mark.slow
def test_train_vit_end_to_end(tmp_path):
    onnx_path = run_experiment_on_digits_dataset(
        tmp_path,
        experiment='train_vit',
        num_classes=2,
        extra_overrides=['model.net.pretrained=false'],
    )
    assert_exported_classifier(onnx_path, class_neurons=2)


@pytest.mark.slow
def test_full_flow_multihead_train_load_test_and_predict(tmp_path):
    """Train -> reload the checkpoint from disk -> trainer.test -> trainer.predict,
    through the real eval task, for the explainability CNN."""
    onnx_path = run_experiment_on_digits_dataset(
        tmp_path,
        experiment='train_cnn_multi',
        num_classes=2,
        extra_overrides=['model.net.weights=null'],
    )
    ckpt_path = onnx_path.replace('.onnx', '.ckpt')
    # eval.py does not auto-inject num_classes like train.py, hence +model.num_classes
    eval_overrides = ['model.net.weights=null', '+model.num_classes=2']

    metrics = run_eval_on_digits_dataset(
        tmp_path, 'train_cnn_multi', ckpt_path, eval_overrides
    )
    assert 0.0 <= float(metrics['test/acc']) <= 1.0

    run_eval_on_digits_dataset(
        tmp_path, 'train_cnn_multi', ckpt_path, eval_overrides, predict=True
    )


@pytest.mark.slow
def test_full_flow_plain_cnn_train_load_test_and_predict(tmp_path):
    """Same full flow for the plain (non-explainability) CNN."""
    onnx_path = run_experiment_on_digits_dataset(
        tmp_path,
        experiment='train_cnn',
        num_classes=2,
        extra_overrides=['model.net.pretrained=false'],
    )
    ckpt_path = onnx_path.replace('.onnx', '.ckpt')
    eval_overrides = ['model.net.pretrained=false', '+model.num_classes=2']

    metrics = run_eval_on_digits_dataset(
        tmp_path, 'train_cnn', ckpt_path, eval_overrides
    )
    assert 0.0 <= float(metrics['test/acc']) <= 1.0

    run_eval_on_digits_dataset(
        tmp_path, 'train_cnn', ckpt_path, eval_overrides, predict=True
    )


@pytest.mark.slow
def test_train_grain_multiclass_end_to_end(tmp_path):
    # also covers the plain multiclass head and the custom
    # ResizeAndExtrapolateBorders transforms
    onnx_path = run_experiment_on_digits_dataset(
        tmp_path,
        experiment='train_grain',
        num_classes=3,
        extra_overrides=['model.net.pretrained=false'],
    )
    assert_exported_classifier(onnx_path, class_neurons=3)
