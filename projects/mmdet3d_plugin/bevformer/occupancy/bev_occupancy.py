"""Utilities for building 2D BEV occupancy targets from 3D boxes.

The mask convention is ego-centric LiDAR/BEV coordinates:

* x points forward in meters and maps to mask columns.
* y points left in meters and maps to mask rows.
* row 0 is y_min and col 0 is x_min.
* visualizations should usually use ``origin="lower"``.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from matplotlib.path import Path


VEHICLE_CLASSES = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
)


@dataclass(frozen=True)
class BEVGridSpec:
    """Metric BEV range and discrete mask shape."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    height: int
    width: int

    def __post_init__(self):
        if not self.x_min < self.x_max:
            raise ValueError(f"expected x_min < x_max, got {self.x_min} >= {self.x_max}")
        if not self.y_min < self.y_max:
            raise ValueError(f"expected y_min < y_max, got {self.y_min} >= {self.y_max}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"expected positive grid shape, got {(self.height, self.width)}")

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def cell_size_x(self):
        return (self.x_max - self.x_min) / float(self.width)

    @property
    def cell_size_y(self):
        return (self.y_max - self.y_min) / float(self.height)

    @property
    def extent(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def metric_to_grid(self, xy):
        """Map metric ``[..., 2]`` xy coordinates to fractional row/col coordinates."""
        xy = np.asarray(xy, dtype=np.float32)
        if xy.shape[-1] != 2:
            raise ValueError(f"expected last coordinate dim 2, got shape {xy.shape}")
        col = (xy[..., 0] - self.x_min) / self.cell_size_x
        row = (xy[..., 1] - self.y_min) / self.cell_size_y
        return np.stack([row, col], axis=-1)

    def cell_centers_in_window(self, row_min, row_max, col_min, col_max):
        rows = np.arange(row_min, row_max + 1, dtype=np.float32)
        cols = np.arange(col_min, col_max + 1, dtype=np.float32)
        y = self.y_min + (rows + 0.5) * self.cell_size_y
        x = self.x_min + (cols + 0.5) * self.cell_size_x
        xx, yy = np.meshgrid(x, y)
        return rows.astype(np.int64), cols.astype(np.int64), np.stack([xx, yy], axis=-1)


def bev_grid_from_config(cfg):
    """Infer the occupancy grid from a BEVFormer config."""
    point_cloud_range = list(cfg.point_cloud_range)
    if len(point_cloud_range) != 6:
        raise ValueError(f"expected 6-value point_cloud_range, got {point_cloud_range}")
    x_min, y_min, _, x_max, y_max, _ = [float(value) for value in point_cloud_range]

    head_cfg = cfg.model.pts_bbox_head
    bev_h = int(head_cfg.bev_h)
    bev_w = int(head_cfg.bev_w)
    return BEVGridSpec(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        height=bev_h,
        width=bev_w,
    )


def _as_numpy_boxes(boxes):
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1) if boxes.size else boxes.reshape(0, 7)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"expected boxes with shape [N, >=7], got {boxes.shape}")
    return boxes


def _as_numpy_labels(labels):
    if labels is None:
        return None
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"expected labels with shape [N], got {labels.shape}")
    return labels


def select_box_indices(labels=None, names=None, class_names=None, selected_classes=VEHICLE_CLASSES):
    """Return indices for boxes whose class is in ``selected_classes``."""
    selected_classes = set(selected_classes)
    if names is not None:
        names = [str(name) for name in names]
        return np.asarray([i for i, name in enumerate(names) if name in selected_classes], dtype=np.int64)

    labels = _as_numpy_labels(labels)
    if labels is None:
        raise ValueError("either labels or names must be provided to select occupancy boxes")
    if class_names is None:
        raise ValueError("class_names is required when selecting by numeric labels")
    class_names = list(class_names)
    keep = []
    for i, label in enumerate(labels.astype(np.int64).tolist()):
        if 0 <= label < len(class_names) and class_names[label] in selected_classes:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def boxes_to_bev_footprints(boxes):
    """Return oriented BEV footprint polygons with shape ``[N, 4, 2]``."""
    boxes = _as_numpy_boxes(boxes)
    footprints = np.zeros((boxes.shape[0], 4, 2), dtype=np.float32)
    for i, box in enumerate(boxes):
        x, y, _, dx, dy, _, yaw = box[:7].tolist()
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError(f"box {i} has non-positive BEV dimensions dx={dx}, dy={dy}")
        hx = dx * 0.5
        hy = dy * 0.5
        local = np.asarray([[hx, hy],
                            [hx, -hy],
                            [-hx, -hy],
                            [-hx, hy]], dtype=np.float32)
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        # p_ego = [x, y] + R(yaw) p_local
        footprints[i] = local @ rot.T + np.asarray([x, y], dtype=np.float32)
    return footprints


def _point_in_rect(point, rect_min, rect_max, eps=1e-6):
    return bool(
        rect_min[0] - eps <= point[0] <= rect_max[0] + eps
        and rect_min[1] - eps <= point[1] <= rect_max[1] + eps
    )


def _orientation(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a, b, c, eps=1e-6):
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
        and abs(_orientation(a, b, c)) <= eps
    )


def _segments_intersect(a, b, c, d, eps=1e-6):
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    return (
        _on_segment(a, c, b, eps)
        or _on_segment(a, d, b, eps)
        or _on_segment(c, a, d, eps)
        or _on_segment(c, b, d, eps)
    )


def _polygon_intersects_cell(polygon, polygon_path, x0, x1, y0, y1):
    rect = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    if polygon_path.contains_points(rect, radius=1e-6).any():
        return True
    rect_min = rect.min(axis=0)
    rect_max = rect.max(axis=0)
    if any(_point_in_rect(point, rect_min, rect_max) for point in polygon):
        return True
    rect_edges = list(zip(rect, np.roll(rect, -1, axis=0)))
    polygon_edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))
    return any(
        _segments_intersect(poly_a, poly_b, rect_a, rect_b)
        for poly_a, poly_b in polygon_edges
        for rect_a, rect_b in rect_edges
    )


def rasterize_bev_occupancy(
    boxes,
    grid,
    labels=None,
    names=None,
    class_names=None,
    selected_classes=VEHICLE_CLASSES,
):
    """Rasterize selected 3D boxes into a binary ``[H, W]`` BEV mask.

    A cell is marked occupied when its metric square intersects a selected box
    footprint. This makes the target a conservative binary coverage mask under
    the stated row/column metric mapping.
    """
    boxes = _as_numpy_boxes(boxes)
    indices = select_box_indices(labels, names, class_names, selected_classes)
    if len(indices) > len(boxes):
        raise AssertionError("selected more boxes than provided")
    selected_boxes = boxes[indices]
    footprints = boxes_to_bev_footprints(selected_boxes)
    mask = np.zeros(grid.shape, dtype=np.uint8)

    for polygon in footprints:
        row_col = grid.metric_to_grid(polygon)
        row_min = max(int(math.floor(float(row_col[:, 0].min()))), 0)
        row_max = min(int(math.ceil(float(row_col[:, 0].max()))), grid.height - 1)
        col_min = max(int(math.floor(float(row_col[:, 1].min()))), 0)
        col_max = min(int(math.ceil(float(row_col[:, 1].max()))), grid.width - 1)
        if row_min > row_max or col_min > col_max:
            continue

        polygon_path = Path(polygon)
        for row in range(row_min, row_max + 1):
            y0 = grid.y_min + row * grid.cell_size_y
            y1 = y0 + grid.cell_size_y
            for col in range(col_min, col_max + 1):
                x0 = grid.x_min + col * grid.cell_size_x
                x1 = x0 + grid.cell_size_x
                if _polygon_intersects_cell(polygon, polygon_path, x0, x1, y0, y1):
                    mask[row, col] = 1

    if mask.shape != grid.shape:
        raise AssertionError(f"unexpected mask shape {mask.shape}, expected {grid.shape}")
    if not np.isin(mask, [0, 1]).all():
        raise AssertionError("occupancy mask must be binary")
    return mask, footprints, indices
