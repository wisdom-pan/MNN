# PP-OCRv6 OCR Engine

End-to-end text detection + recognition for [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
PP-OCRv6 models, built on MNN. Detection uses the DB algorithm; recognition uses
a CTC-decoded CRNN/SVTR head. All image operations use MNN's built-in
OpenCV-compatible module (`tools/cv`), so there is no external dependency.

## Build

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_OCR=ON && make ocr_demo -j$(nproc)
```

`MNN_BUILD_OCR` automatically enables `MNN_BUILD_OPENCV` and `MNN_IMGCODECS`.

## Run

```bash
./ocr_demo <det.mnn> <rec.mnn> <keys.txt> <image> [backend=0(CPU)]
```

Example:

```bash
./ocr_demo PP-OCRv6_medium_det.mnn PP-OCRv6_medium_rec.mnn ppocr_keys_v6_medium.txt sample.jpg
```

## Pipeline

| Stage | Detail |
|-------|--------|
| Det preprocess | Resize long side to `det_limit_side_len` (default 960), round to a multiple of 32; ImageNet normalization `(x/255 - mean)/std`. |
| Det postprocess (DB) | Threshold the probability map (`det_db_thresh`), `findContours` → `minAreaRect` → box-score filter (`det_db_box_thresh`) → unclip (`det_db_unclip_ratio`). |
| Rec preprocess | Perspective-crop each box, resize to height `rec_img_height` (default 48), normalize to `[-1, 1]`. |
| Rec postprocess (CTC) | Greedy decode: argmax per timestep, drop blank (index 0), collapse repeats, map index → character. |

## Model repository layout (for the Android model market)

Each downloadable variant is a directory containing det/rec models, the key
dictionary, and an `ocr_config.json` descriptor:

```
PP-OCRv6-medium-MNN/
├── det.mnn
├── rec.mnn
├── keys.txt
└── ocr_config.json
```

`ocr_config.json` (all fields optional; defaults shown):

```json
{
    "det_model": "det.mnn",
    "rec_model": "rec.mnn",
    "keys": "keys.txt",
    "det_limit_side_len": 960,
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.5,
    "det_db_unclip_ratio": 1.5,
    "rec_img_height": 48
}
```

The recognition character list is `[blank] + <dictionary lines> + [space]`, so the
model vocabulary size equals `num_keys + 2`.
