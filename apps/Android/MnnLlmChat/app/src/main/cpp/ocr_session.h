//
// Created by MNN on 2026/06/25.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ocr/ocr.hpp"

namespace mls {

// Thin wrapper around MNN::OCR::Ocr. Loads det/rec models and a key dictionary
// from a model directory containing an ocr_config.json descriptor.
class OcrSession {
public:
    OcrSession(std::string resource_path, int backend_type);

    // Load the underlying models. Returns false on failure.
    bool Load();

    // Run OCR on an image file, returning all recognized text regions.
    std::vector<MNN::OCR::OcrResult> Run(const std::string& image_path);

private:
    std::string resource_path_;
    int backend_type_;
    bool loaded_{false};
    std::unique_ptr<MNN::OCR::Ocr> ocr_{nullptr};
};

}  // namespace mls
