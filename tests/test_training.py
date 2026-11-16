import pathlib
import sys
import torch
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.student import StudentConfig, StudentSLM, build_student_slm
from src.models.distillation import DistillationLoss, FocalLoss
from src.training.trainer import TrainConfig, StanceDataset, LABEL2ID, ID2LABEL


def test_label_mapping_consistency():
    assert len(LABEL2ID) == 3
    for lbl, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == lbl


def test_student_config_defaults():
    cfg = StudentConfig()
    assert cfg.num_hidden_layers == 4
    assert cfg.num_labels == 3
    assert cfg.pooling == "mean"


def test_distillation_loss_forward():
    criterion = DistillationLoss(temperature=2.0, alpha=0.5, beta=0.5)
    s_logits = torch.randn(4, 3)
    t_logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])

    loss_dict = criterion(s_logits, t_logits, targets)
    assert "loss" in loss_dict
    assert "ce" in loss_dict
    assert "kl" in loss_dict
    assert loss_dict["loss"].item() > 0.0


def test_focal_loss_forward():
    focal = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    loss = focal(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
