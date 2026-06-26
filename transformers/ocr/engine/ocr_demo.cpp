//
//  ocr_demo.cpp
//
//  Created by MNN on 2026/06/25.
//  MNN
//
//  CLI harness for the PP-OCRv6 OCR engine.
//  Usage: ./ocr_demo <det.mnn> <rec.mnn> <keys.txt> <image> [backend]
//
#include <iostream>

#include <MNN/AutoTime.hpp>

#include "ocr/ocr.hpp"

using namespace MNN::OCR;

int main(int argc, const char* argv[]) {
    if (argc < 5) {
        MNN_PRINT("Usage: ./ocr_demo <det.mnn> <rec.mnn> <keys.txt> <image> [backend=0(CPU)]\n");
        return 0;
    }
    OcrConfig config;
    config.detModel = argv[1];
    config.recModel = argv[2];
    config.keysFile = argv[3];
    std::string imagePath = argv[4];
    if (argc >= 6) {
        config.backendType = (MNNForwardType)atoi(argv[5]);
    }

    Ocr ocr(config);
    if (!ocr.load()) {
        MNN_ERROR("Failed to load OCR models.\n");
        return 1;
    }

    std::vector<OcrResult> results;
    {
        AUTOTIME;
        results = ocr.run(imagePath);
    }

    MNN_PRINT("Detected %d text regions:\n", (int)results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        MNN_PRINT("  [%2d] box=[", (int)i);
        for (size_t j = 0; j < r.box.size(); ++j) {
            MNN_PRINT("(%d,%d)", r.box[j].first, r.box[j].second);
            if (j + 1 < r.box.size()) {
                MNN_PRINT(",");
            }
        }
        MNN_PRINT("] score=%.3f text=\"%s\"\n", r.score, r.text.c_str());
    }
    return 0;
}
