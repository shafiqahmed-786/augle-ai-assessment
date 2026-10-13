"""Geometry correctness: letterbox, flips, affine transforms, masks, normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from augle.config import DataConfig
from dataset import (
    DetectionTransform,
    Sample,
    apply_affine,
    box_areas,
    build_affine,
    clip_boxes,
    denormalize_to_uint8,
    flip_horizontal,
    flip_vertical,
    letterbox,
    mask_tight_boxes,
    normalize_image,
    segmentation_to_mask,
    to_opencv_matrix,
    transform_boxes,
)


def rect_sample(h: int = 100, w: int = 200, box=(40, 20, 120, 60)) -> Sample:
    """Black image with one white filled rectangle; the mask is exactly the box."""
    image = np.zeros((h, w, 3), np.uint8)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = 255
    mask = np.zeros((1, h, w), np.uint8)
    mask[0, y1:y2, x1:x2] = 1
    return Sample(image, np.array([box], np.float32), np.array([0]), mask)


def test_identity_affine_is_identity():
    m = build_affine(64, 48)
    assert np.allclose(m, np.eye(3))
    assert np.allclose(to_opencv_matrix(m), np.eye(3)[:2])
    boxes = np.array([[3, 4, 20, 30]], np.float32)
    assert np.allclose(transform_boxes(boxes, m), boxes)


def test_letterbox_scales_and_pads_exactly():
    s = rect_sample(100, 200, (40, 20, 120, 60))
    out, info = letterbox(s, 100)
    assert out.image.shape == (100, 100, 3)
    assert (info.pad_x, info.pad_y) == (0, 25)
    assert np.allclose(out.boxes[0], [20, 35, 60, 55])  # x,y scaled by 0.5, y shifted by pad 25
    assert np.array_equal(mask_tight_boxes(out.masks)[0], out.boxes[0])
    # pixels under the box are white, a pixel far outside is the pad colour
    assert out.image[45, 40].tolist() == [255, 255, 255]
    assert out.image[2, 2].tolist() == [114, 114, 114]


def test_letterbox_portrait_and_upscale():
    s = rect_sample(200, 100, (10, 50, 60, 150))
    out, info = letterbox(s, 400)  # upscale x2
    assert (info.pad_x, info.pad_y) == (100, 0)
    assert np.allclose(out.boxes[0], [120, 100, 220, 300])
    assert np.array_equal(mask_tight_boxes(out.masks)[0], out.boxes[0])


def test_flips_match_mask_derived_boxes():
    s = rect_sample()
    h = flip_horizontal(s)
    v = flip_vertical(s)
    assert np.allclose(h.boxes[0], [200 - 120, 20, 200 - 40, 60])
    assert np.allclose(v.boxes[0], [40, 100 - 60, 120, 100 - 20])
    assert np.array_equal(mask_tight_boxes(h.masks)[0], h.boxes[0])
    assert np.array_equal(mask_tight_boxes(v.masks)[0], v.boxes[0])
    # double flip is the identity for image, boxes and masks
    hh = flip_horizontal(h)
    assert np.array_equal(hh.image, s.image) and np.allclose(hh.boxes, s.boxes) and np.array_equal(hh.masks, s.masks)


def test_integer_translation_shifts_box_and_mask_exactly():
    s = rect_sample()
    m = build_affine(200, 100, tx=10, ty=5)
    out = apply_affine(s, m)
    assert np.allclose(out.boxes[0], [50, 25, 130, 65])
    assert np.array_equal(mask_tight_boxes(out.masks)[0], out.boxes[0])
    assert out.image[30, 60].tolist() == [255, 255, 255] and out.image[30, 45].tolist() == [0, 0, 0]


def test_rotation_envelope_matches_analytic_formula():
    # box centred on the canvas centre: envelope of a rotated w x h rectangle
    w, h, ang = 40.0, 20.0, 30.0
    box = np.array([[100 - w / 2, 100 - h / 2, 100 + w / 2, 100 + h / 2]], np.float32)
    out = transform_boxes(box, build_affine(200, 200, angle_deg=ang))[0]
    a = np.deg2rad(ang)
    exp_w, exp_h = w * np.cos(a) + h * np.sin(a), w * np.sin(a) + h * np.cos(a)
    assert out[2] - out[0] == pytest.approx(exp_w, abs=1e-3)
    assert out[3] - out[1] == pytest.approx(exp_h, abs=1e-3)
    assert (out[0] + out[2]) / 2 == pytest.approx(100, abs=1e-3)  # still centred


@pytest.mark.parametrize(
    "angle,scale,shear_x,shear_y,tx,ty",
    [
        (0, 1.0, 0, 0, 0, 0),
        (15, 1.0, 0, 0, 0, 0),
        (-25, 1.0, 0, 0, 0, 0),
        (30, 0.8, 0, 0, 12, -7),
        (-10, 1.2, 5, -4, -9, 6),
        (45, 1.0, 0, 0, 0, 0),
        (0, 1.0, 10, 0, 0, 0),
    ],
)
def test_image_mask_and_box_stay_aligned_under_affine(angle, scale, shear_x, shear_y, tx, ty):
    """The tight box of the *warped mask* must equal the corner envelope of the transformed box.

    This pins down rotation sign, shear, scale, translation and the half-pixel convention at once.
    """
    s = rect_sample(200, 200, (70, 80, 130, 110))
    m = build_affine(200, 200, angle, scale, shear_x, shear_y, tx, ty)
    envelope = transform_boxes(s.boxes, m)[0]
    out = apply_affine(s, m, refine_with_masks=False, min_visibility=0.0, min_box_side=0.0)
    mask_box = mask_tight_boxes(out.masks)[0]
    assert np.max(np.abs(mask_box - envelope)) <= 1.5, (mask_box, envelope)
    # mask-derived tight box never exceeds the envelope (beyond 1px rasterisation slack)
    assert mask_box[0] >= envelope[0] - 1.5 and mask_box[2] <= envelope[2] + 1.5
    # image content moved with the mask: the white pixels' bounding box agrees too
    ys, xs = np.nonzero(out.image[..., 0] > 200)
    assert abs(xs.min() - mask_box[0]) <= 2 and abs(xs.max() + 1 - mask_box[2]) <= 2
    assert abs(ys.min() - mask_box[1]) <= 2 and abs(ys.max() + 1 - mask_box[3]) <= 2


def test_rotation_is_not_plain_scaling():
    """A rotated box must grow (envelope), not stay the same size."""
    box = np.array([[80, 90, 120, 110]], np.float32)
    rotated = transform_boxes(box, build_affine(200, 200, angle_deg=40))[0]
    assert (rotated[2] - rotated[0]) > 40 and (rotated[3] - rotated[1]) > 20


def test_refine_with_masks_gives_tight_box_inside_envelope():
    s = rect_sample(200, 200, (70, 80, 130, 110))
    m = build_affine(200, 200, angle_deg=30)
    loose = apply_affine(s, m, refine_with_masks=False)
    tight = apply_affine(s, m, refine_with_masks=True)
    assert box_areas(tight.boxes)[0] <= box_areas(loose.boxes)[0] + 1e-6
    assert np.array_equal(mask_tight_boxes(tight.masks)[0], tight.boxes[0])


def test_out_of_frame_instances_are_dropped_consistently():
    s = rect_sample()
    out = apply_affine(s, build_affine(200, 100, tx=500))
    assert len(out.boxes) == len(out.labels) == 0
    assert out.masks.shape == (0, 100, 200)


def test_partial_visibility_clips_box_to_canvas_and_respects_threshold():
    s = rect_sample(100, 200, (150, 20, 190, 60))
    m = build_affine(200, 100, tx=30)  # right half of the object leaves the canvas
    kept = apply_affine(s, m, min_visibility=0.3, refine_with_masks=False)
    assert len(kept.boxes) == 1
    assert kept.boxes[0][2] == pytest.approx(200.0) and kept.boxes[0][0] == pytest.approx(180.0)
    dropped = apply_affine(s, m, min_visibility=0.9, refine_with_masks=False)
    assert len(dropped.boxes) == 0


def test_clip_boxes_and_degenerate_areas():
    clipped = clip_boxes(np.array([[-5, -5, 300, 120], [10, 10, 10, 20]], np.float32), 200, 100)
    assert np.allclose(clipped[0], [0, 0, 200, 100])
    assert box_areas(clipped)[1] == 0  # zero-width box has zero area


def test_empty_sample_passes_through_every_transform():
    empty = Sample(
        np.full((60, 80, 3), 7, np.uint8), np.zeros((0, 4)), np.zeros((0,), np.int64), np.zeros((0, 60, 80), np.uint8)
    )
    out, _ = letterbox(empty, 64)
    assert out.boxes.shape == (0, 4) and out.masks.shape == (0, 64, 64)
    out = flip_horizontal(flip_vertical(out))
    out = apply_affine(out, build_affine(64, 64, angle_deg=20, scale=1.1))
    assert out.image.shape == (64, 64, 3) and len(out.boxes) == 0


def test_full_train_pipeline_keeps_boxes_equal_to_mask_boxes():
    cfg = DataConfig(image_size=128, rotation_deg=20, shear_deg=8, vflip_p=0.5)
    tf = DetectionTransform(128, True, cfg)
    base = rect_sample(100, 200, (40, 20, 120, 60))
    checked = 0
    for seed in range(40):
        out = tf(base, np.random.default_rng(seed))
        assert out.image.shape == (128, 128, 3) and out.image.dtype == np.uint8
        assert len(out.boxes) == len(out.labels) == len(out.masks)
        for box, mask in zip(out.boxes, out.masks):
            assert np.array_equal(mask_tight_boxes(mask[None])[0], box)
            assert 0 <= box[0] < box[2] <= 128 and 0 <= box[1] < box[3] <= 128
            checked += 1
    assert checked > 20


def test_pipeline_is_deterministic_given_rng():
    cfg = DataConfig(image_size=96)
    tf = DetectionTransform(96, True, cfg)
    a = tf(rect_sample(), np.random.default_rng([1, 2, 3]))
    b = tf(rect_sample(), np.random.default_rng([1, 2, 3]))
    assert np.array_equal(a.image, b.image) and np.array_equal(a.boxes, b.boxes)


def test_eval_transform_has_no_randomness():
    tf = DetectionTransform(64, False, DataConfig())
    a, b = tf(rect_sample(), np.random.default_rng(0)), tf(rect_sample(), np.random.default_rng(99))
    assert np.array_equal(a.image, b.image) and np.array_equal(a.boxes, b.boxes)


def test_normalisation_roundtrip_and_shape():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    arr = normalize_image(img, mean, std)
    assert arr.shape == (3, 16, 24) and arr.dtype == np.float32
    import torch

    back = denormalize_to_uint8(torch.from_numpy(arr), mean, std)
    assert np.max(np.abs(back.astype(int) - img.astype(int))) <= 1


def test_polygon_rasterisation_matches_mask_area():
    poly = [[10.0, 10.0, 30.0, 10.0, 30.0, 25.0, 10.0, 25.0]]
    mask = segmentation_to_mask(poly, 40, 50)
    assert mask.shape == (40, 50)
    # continuous (pixel-edge) coordinates: pixels whose centres lie inside -> columns 10..29, rows 10..24
    assert mask_tight_boxes(mask[None])[0].tolist() == [10, 10, 30, 25]
    assert int(mask.sum()) == 20 * 15
    assert segmentation_to_mask([], 40, 50) is None


def test_uncompressed_rle_decodes_column_major():
    # 3 rows x 4 cols; column-major runs: 3 zeros, 3 ones (column 1), rest zeros
    rle = {"size": [3, 4], "counts": [3, 3, 6]}
    mask = segmentation_to_mask(rle, 3, 4)
    assert mask.shape == (3, 4)
    assert mask[:, 1].tolist() == [1, 1, 1] and mask.sum() == 3
