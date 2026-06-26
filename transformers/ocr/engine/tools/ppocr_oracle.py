#!/usr/bin/env python3
"""PP-OCRv6 reference pipeline (det + DB postprocess + rec + CTC decode).

This is an *oracle* used while implementing the C++ engine: it pins down the
exact preprocessing constants, DB postprocessing, and CTC decode against the
real .mnn files so the C++ output can be validated against it. It deliberately
uses MNN.cv (the same OpenCV-compatible primitives the C++ engine uses).

Usage:
    python3 ppocr_oracle.py <det.mnn> <rec.mnn> <keys.txt> <image>
"""
import sys
import math
import numpy as np
import MNN
import MNN.cv as cv
import MNN.expr as expr


def load_keys(path):
    # PaddleOCR char list: index 0 = CTC blank, then dict lines, then space.
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    return ["[blank]"] + lines + [" "]


def to_np(var):
    return np.array(var.read())


def cv_resize_uint8(var_or_np, dst_w, dst_h):
    """Resize keeping uint8 semantics. MNN.cv.resize corrupts float Vars, so we
    always resize as uint8 and return a float32 numpy array."""
    if isinstance(var_or_np, np.ndarray):
        v = expr.const(var_or_np.astype(np.uint8), list(var_or_np.shape),
                       expr.NHWC, expr.uint8)
    else:
        v = var_or_np
    return to_np(cv.resize(v, (dst_w, dst_h))).astype(np.float32)


def run_module(model_path, np_input):
    """Run a single-input single-output MNN model (NCHW float), return numpy."""
    interp = MNN.Interpreter(model_path)
    sess = interp.createSession({})
    inp = interp.getSessionInput(sess)
    shape = tuple(int(x) for x in np_input.shape)
    interp.resizeTensor(inp, shape)
    interp.resizeSession(sess)
    tmp = MNN.Tensor(shape, MNN.Halide_Type_Float,
                     np.ascontiguousarray(np_input, dtype=np.float32),
                     MNN.Tensor_DimensionType_Caffe)
    inp.copyFrom(tmp)
    interp.runSession(sess)
    out = interp.getSessionOutput(sess)
    out_shape = out.getShape()
    out_host = MNN.Tensor(out_shape, MNN.Halide_Type_Float,
                          np.zeros(out_shape, dtype=np.float32),
                          MNN.Tensor_DimensionType_Caffe)
    out.copyToHostTensor(out_host)
    return np.array(out_host.getNumpyData(), dtype=np.float32).reshape(out_shape)


# ----------------------------- Detection -----------------------------
# PP-OCRv6 det uses ImageNet normalization (canonical PaddleOCR det config).
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def det_preprocess(rgb_np, limit_side_len=960):
    h, w = rgb_np.shape[:2]
    ratio = 1.0
    if max(h, w) > limit_side_len:
        ratio = limit_side_len / max(h, w)
    rh = max(32, int(round(h * ratio / 32) * 32))
    rw = max(32, int(round(w * ratio / 32) * 32))
    resized = cv_resize_uint8(rgb_np, rw, rh)  # [rh, rw, 3] float
    arr = resized / 255.0
    arr = (arr - DET_MEAN) / DET_STD
    arr = arr.transpose(2, 0, 1)[None]  # NCHW
    return np.ascontiguousarray(arr, dtype=np.float32), (w / rw, h / rh)


def poly_perimeter(box):
    p = 0.0
    n = len(box)
    for i in range(n):
        j = (i + 1) % n
        p += math.hypot(box[i][0] - box[j][0], box[i][1] - box[j][1])
    return p


def unclip(box, ratio=1.5):
    # Expand polygon outward by area*ratio/perimeter (Vatti-style offset),
    # approximated by pushing each vertex away from the centroid.
    area = abs(cv.contourArea(expr.const(box.astype(np.float32), [4, 1, 2], expr.NHWC)))
    perimeter = poly_perimeter(box)
    if perimeter < 1e-6:
        return box
    distance = area * ratio / perimeter
    cx, cy = box[:, 0].mean(), box[:, 1].mean()
    out = []
    for x, y in box:
        vx, vy = x - cx, y - cy
        norm = math.hypot(vx, vy) + 1e-6
        out.append([x + vx / norm * distance, y + vy / norm * distance])
    return np.array(out, dtype=np.float32)


def order_points(pts):
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def db_postprocess(prob_map, scale_xy, thresh=0.3, min_size=3):
    sx, sy = scale_xy
    binary = (prob_map > thresh).astype(np.uint8) * 255  # [H,W]
    bvar = expr.const(binary, [binary.shape[0], binary.shape[1], 1], expr.NHWC, expr.uint8)
    contours, _ = cv.findContours(bvar, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        cnp = to_np(c).reshape(-1, 2).astype(np.float32)
        if cnp.shape[0] < 4:
            continue
        rect = cv.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        if min(rw, rh) < min_size:
            continue
        pts = to_np(cv.boxPoints(rect)).reshape(4, 2).astype(np.float32)
        pts = unclip(pts, 1.5)
        pts[:, 0] = np.clip(pts[:, 0] * sx, 0, None)
        pts[:, 1] = np.clip(pts[:, 1] * sy, 0, None)
        boxes.append(order_points(pts))
    boxes.sort(key=lambda b: (round(b[:, 1].min() / 10.0), b[:, 0].min()))
    return boxes


def crop_box(rgb_np, box):
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv.getPerspectiveTransform(box.astype(np.float32), dst)
    var = expr.const(rgb_np.astype(np.uint8), list(rgb_np.shape), expr.NHWC, expr.uint8)
    crop = to_np(cv.warpPerspective(var, M, (w, h))).astype(np.float32)
    if h * 1.0 / w >= 1.5:
        crop = np.rot90(crop)
    return crop


# ----------------------------- Recognition -----------------------------
def rec_preprocess(crop_np, img_h=48):
    h, w = crop_np.shape[:2]
    rw = max(1, int(math.ceil(img_h * w / float(h))))
    resized = cv_resize_uint8(crop_np, rw, img_h)
    arr = resized / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def ctc_decode(logits, keys):
    preds = logits[0]  # (T, V)
    idx = preds.argmax(axis=1)
    prob = preds.max(axis=1)
    chars, scores, last = [], [], -1
    for t, i in enumerate(idx):
        if i != 0 and i != last:  # 0 = blank; collapse repeats
            if i < len(keys):
                chars.append(keys[i])
                scores.append(prob[t])
        last = i
    return "".join(chars), (float(np.mean(scores)) if scores else 0.0)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    det_path, rec_path, keys_path, img_path = sys.argv[1:5]
    keys = load_keys(keys_path)
    print(f"vocab size (blank+keys+space) = {len(keys)}")

    bgr = cv.imread(img_path)
    rgb = to_np(cv.cvtColor(bgr, cv.COLOR_BGR2RGB)).astype(np.float32)
    print(f"image shape = {rgb.shape}")

    det_in, scale_xy = det_preprocess(rgb)
    print(f"det input = {det_in.shape}, scale = {scale_xy}")
    prob = run_module(det_path, det_in)
    print(f"det output = {prob.shape}, prob range [{prob.min():.3f}, {prob.max():.3f}]")
    boxes = db_postprocess(prob[0, 0], scale_xy)
    print(f"detected {len(boxes)} boxes")

    for bi, box in enumerate(boxes):
        crop = crop_box(rgb, box)
        rec_in = rec_preprocess(crop)
        logits = run_module(rec_path, rec_in)
        text, score = ctc_decode(logits, keys)
        bb = [[int(x), int(y)] for x, y in box]
        print(f"  box[{bi}] {bb} -> '{text}' (score={score:.3f})")


if __name__ == "__main__":
    main()
