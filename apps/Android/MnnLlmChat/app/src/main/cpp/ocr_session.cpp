//
// Created by MNN on 2026/06/25.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.
//

#include "ocr_session.h"

#include <fstream>
#include <sstream>
#include <utility>

#include "mls_log.h"
#include "nlohmann/json.hpp"

namespace mls {

using json = nlohmann::json;

namespace {
std::string joinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}
}  // namespace

OcrSession::OcrSession(std::string resource_path, int backend_type)
    : resource_path_(std::move(resource_path)), backend_type_(backend_type) {}

bool OcrSession::Load() {
    if (loaded_) {
        return true;
    }
    // Read ocr_config.json from the model directory; fall back to defaults.
    std::string configPath = joinPath(resource_path_, "ocr_config.json");
    std::string detName = "det.mnn";
    std::string recName = "rec.mnn";
    std::string keysName = "keys.txt";
    MNN::OCR::OcrConfig config;
    std::ifstream in(configPath);
    if (in.good()) {
        std::stringstream ss;
        ss << in.rdbuf();
        try {
            json j = json::parse(ss.str());
            if (j.contains("det_model")) detName = j["det_model"].get<std::string>();
            if (j.contains("rec_model")) recName = j["rec_model"].get<std::string>();
            if (j.contains("keys")) keysName = j["keys"].get<std::string>();
            if (j.contains("det_limit_side_len")) config.detLimitSideLen = j["det_limit_side_len"];
            if (j.contains("det_db_thresh")) config.detDbThresh = j["det_db_thresh"];
            if (j.contains("det_db_box_thresh")) config.detDbBoxThresh = j["det_db_box_thresh"];
            if (j.contains("det_db_unclip_ratio")) config.detDbUnclipRatio = j["det_db_unclip_ratio"];
            if (j.contains("rec_img_height")) config.recImgHeight = j["rec_img_height"];
        } catch (const std::exception& e) {
            MNN_ERROR("OcrSession: failed to parse %s: %s\n", configPath.c_str(), e.what());
        }
    } else {
        MNN_DEBUG("OcrSession: no ocr_config.json at %s, using defaults\n", configPath.c_str());
    }

    config.detModel = joinPath(resource_path_, detName);
    config.recModel = joinPath(resource_path_, recName);
    config.keysFile = joinPath(resource_path_, keysName);
    config.backendType = static_cast<MNNForwardType>(backend_type_);

    ocr_.reset(new MNN::OCR::Ocr(config));
    loaded_ = ocr_->load();
    if (!loaded_) {
        MNN_ERROR("OcrSession: failed to load OCR models from %s\n", resource_path_.c_str());
    }
    return loaded_;
}

std::vector<MNN::OCR::OcrResult> OcrSession::Run(const std::string& image_path) {
    if (!loaded_ && !Load()) {
        return {};
    }
    return ocr_->run(image_path);
}

}  // namespace mls
