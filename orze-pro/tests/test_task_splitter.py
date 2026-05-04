"""Test task_splitter detect_tasks and propose_gpu_allocation."""

import sys
import os

# Ensure source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Bypass license gate for testing
os.environ["ORZE_PRO_LICENSE_KEY"] = "test"
import orze_pro._gate
orze_pro._gate.require_license = lambda: None

from orze_pro.agents.task_splitter import detect_tasks, propose_gpu_allocation, format_proposal


MULTI_TASK_GOAL = """\
# Research Goal

## Tasks

### Task 1: Image Classification
- **Dataset**: CIFAR-10
- **Metric**: test_accuracy
- **Directions**: ResNet, EfficientNet, ViT

### Task 2: Object Detection
- **Dataset**: COCO
- **Metric**: mAP
- **Directions**: YOLO, DETR, Faster R-CNN
"""

SINGLE_TASK_GOAL = """\
# Research Goal

Train a model on CIFAR-10 to maximize test accuracy.
"""

NO_TASKS_GOAL = """\
# Notes

Just some random notes, no task structure here.
"""


def test_multi_task_detection():
    tasks = detect_tasks(MULTI_TASK_GOAL)
    assert len(tasks) == 2, f"Expected 2 tasks, got {len(tasks)}: {tasks}"

    assert tasks[0]["title"] == "Image Classification"
    assert tasks[0]["metric"] == "test_accuracy"
    assert tasks[0]["sort"] == "descending"
    assert tasks[0]["name"] == "image_classification"
    assert "ResNet" in tasks[0]["directions"]

    assert tasks[1]["title"] == "Object Detection"
    assert tasks[1]["metric"] == "mAP"
    assert tasks[1]["sort"] == "descending"
    assert tasks[1]["name"] == "object_detection"
    assert "YOLO" in tasks[1]["directions"]

    print("PASS: multi_task_detection")


def test_single_task_detection():
    tasks = detect_tasks(SINGLE_TASK_GOAL)
    assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
    assert tasks[0]["name"] == "main"
    print("PASS: single_task_detection")


def test_no_tasks():
    tasks = detect_tasks(NO_TASKS_GOAL)
    assert len(tasks) == 1
    assert tasks[0]["name"] == "main"
    print("PASS: no_tasks")


def test_gpu_allocation_8gpus():
    tasks = detect_tasks(MULTI_TASK_GOAL)
    tasks = propose_gpu_allocation(tasks, 8)
    total = sum(len(t["gpus"]) for t in tasks)
    assert total == 8, f"Expected 8 GPUs allocated, got {total}"
    for t in tasks:
        assert len(t["gpus"]) >= 1, f"Task {t['name']} got 0 GPUs"
    print(f"PASS: gpu_allocation_8gpus — {[len(t['gpus']) for t in tasks]}")


def test_gpu_allocation_fewer_than_tasks():
    tasks = detect_tasks(MULTI_TASK_GOAL)
    tasks = propose_gpu_allocation(tasks, 1)
    # Both tasks should still get assigned (sharing GPU 0)
    for t in tasks:
        assert len(t["gpus"]) >= 1
    print(f"PASS: gpu_allocation_fewer — {[t['gpus'] for t in tasks]}")


def test_format_proposal():
    tasks = detect_tasks(MULTI_TASK_GOAL)
    tasks = propose_gpu_allocation(tasks, 4)
    from pathlib import Path
    proposal = format_proposal(tasks, Path("/tmp/test"))
    assert "Found 2 tasks" in proposal
    assert "image_classification" in proposal
    assert "object_detection" in proposal
    assert "Proceed?" in proposal
    print("PASS: format_proposal")
    print(proposal)


if __name__ == "__main__":
    test_multi_task_detection()
    test_single_task_detection()
    test_no_tasks()
    test_gpu_allocation_8gpus()
    test_gpu_allocation_fewer_than_tasks()
    test_format_proposal()
    print("\nAll tests passed.")
