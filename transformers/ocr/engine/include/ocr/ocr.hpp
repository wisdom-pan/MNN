//
//  ocr.hpp
//
//  Created by MNN on 2026/06/25.
//  MNN
//
#ifndef MNN_OCR_HPP
#define MNN_OCR_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <MNN/MNNForwardType.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>

namespace MNN {
namespace OCR {

// A single detected + recognized text region.
struct OcrResult {
    // Quadrilateral box in original-image pixel coordinates,
    // ordered top-left, top-right, bottom-right, bottom-left.
    std::vector<std::pair<int, int>> box;
    std::string text;
    float score = 0.f;
};

// Tunable pipeline parameters (defaults match the validated PP-OCRv6 config).
struct OcrConfig {
    std::string detModel;            // path to *_det.mnn
    std::string recModel;            // path to *_rec.mnn
    std::string keysFile;            // path to ppocr_keys_*.txt
    int detLimitSideLen = 960;       // det input long-side cap (multiple of 32)
    float detDbThresh = 0.3f;        // binarization threshold on the prob map
    float detDbBoxThresh = 0.5f;     // per-box mean-score filter
    float detDbUnclipRatio = 1.5f;   // box expansion ratio
    int recImgHeight = 48;           // recognition input height
    MNNForwardType backendType = MNN_FORWARD_CPU;
    int numThread = 4;
};

class MNN_PUBLIC Ocr {
public:
    explicit Ocr(const OcrConfig& config);
    ~Ocr();

    // Load both det and rec models. Returns false on failure.
    bool load();

    // Run the full det -> rec pipeline on an image file.
    std::vector<OcrResult> run(const std::string& imagePath);

    // Convenience: concatenate recognized text lines with '\n'.
    std::string runToText(const std::string& imagePath);

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

}  // namespace OCR
}  // namespace MNN

#endif  // MNN_OCR_HPP
