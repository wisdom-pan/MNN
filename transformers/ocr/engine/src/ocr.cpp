//
//  ocr.cpp
//
//  Created by MNN on 2026/06/25.
//  MNN
//
//  End-to-end PP-OCRv6 pipeline: DB text detection + CRNN/SVTR CTC recognition.
//  Image operations use MNN's built-in OpenCV-compatible module (tools/cv), so
//  there is no external dependency.
//
#include "ocr/ocr.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include "cv/cv.hpp"

namespace MNN {
namespace OCR {

using namespace MNN::Express;

// PP-OCRv6 detection uses ImageNet normalization; recognition maps to [-1, 1].
// cv::resize applies (x - mean) * normal per channel while reading uint8 input,
// which both fuses normalization and avoids float-Var resize artifacts.
static const std::vector<float> kDetMean = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
static const std::vector<float> kDetNorm = {1.f / (0.229f * 255.f), 1.f / (0.224f * 255.f),
                                            1.f / (0.225f * 255.f)};
static const std::vector<float> kRecMean = {127.5f, 127.5f, 127.5f};
static const std::vector<float> kRecNorm = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

namespace {

struct Quad {
    // Four points in clockwise order: tl, tr, br, bl.
    float x[4];
    float y[4];
};

std::vector<std::string> loadKeys(const std::string& path) {
    // PaddleOCR char list: index 0 = CTC blank, then dict lines, then a space.
    std::vector<std::string> keys;
    keys.emplace_back("");  // blank at index 0
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        keys.push_back(line);
    }
    keys.emplace_back(" ");  // trailing space token
    return keys;
}

float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Expand a quad outward (DB "unclip") by area * ratio / perimeter, approximated
// by offsetting each vertex away from the centroid.
Quad unclip(const Quad& q, float ratio) {
    float area = 0.f, perimeter = 0.f;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) & 3;
        area += q.x[i] * q.y[j] - q.x[j] * q.y[i];
        perimeter += std::hypot(q.x[i] - q.x[j], q.y[i] - q.y[j]);
    }
    area = std::fabs(area) * 0.5f;
    Quad out = q;
    if (perimeter < 1e-6f) {
        return out;
    }
    float distance = area * ratio / perimeter;
    float cx = (q.x[0] + q.x[1] + q.x[2] + q.x[3]) * 0.25f;
    float cy = (q.y[0] + q.y[1] + q.y[2] + q.y[3]) * 0.25f;
    for (int i = 0; i < 4; ++i) {
        float vx = q.x[i] - cx, vy = q.y[i] - cy;
        float norm = std::hypot(vx, vy) + 1e-6f;
        out.x[i] = q.x[i] + vx / norm * distance;
        out.y[i] = q.y[i] + vy / norm * distance;
    }
    return out;
}

// Reorder four points to tl, tr, br, bl.
Quad orderPoints(const Quad& in) {
    int order[4] = {0, 1, 2, 3};
    // sort by (x + y) -> tl smallest, br largest; by (x - y) -> tr largest, bl smallest
    int tl = 0, br = 0, tr = 0, bl = 0;
    float sMin = 1e18f, sMax = -1e18f, dMin = 1e18f, dMax = -1e18f;
    for (int i = 0; i < 4; ++i) {
        float s = in.x[i] + in.y[i];
        float d = in.x[i] - in.y[i];
        if (s < sMin) { sMin = s; tl = i; }
        if (s > sMax) { sMax = s; br = i; }
        if (d > dMax) { dMax = d; tr = i; }
        if (d < dMin) { dMin = d; bl = i; }
    }
    (void)order;
    Quad out;
    int idx[4] = {tl, tr, br, bl};
    for (int i = 0; i < 4; ++i) {
        out.x[i] = in.x[idx[i]];
        out.y[i] = in.y[idx[i]];
    }
    return out;
}

}  // namespace

class Ocr::Impl {
public:
    explicit Impl(const OcrConfig& config) : mConfig(config) {}

    bool load() {
        mKeys = loadKeys(mConfig.keysFile);
        if (mKeys.size() <= 2) {
            MNN_ERROR("OCR: failed to load keys from %s\n", mConfig.keysFile.c_str());
            return false;
        }
        ScheduleConfig sconfig;
        sconfig.type = mConfig.backendType;
        sconfig.numThread = mConfig.numThread;
        BackendConfig bconfig;
        bconfig.precision = BackendConfig::Precision_High;
        sconfig.backendConfig = &bconfig;
        mRuntime.reset(Executor::RuntimeManager::createRuntimeManager(sconfig));

        Module::Config mconfig;
        mconfig.shapeMutable = true;  // det/rec accept variable spatial sizes
        mDet.reset(Module::load({"x"}, {"fetch_name_0"}, mConfig.detModel.c_str(), mRuntime, &mconfig));
        mRec.reset(Module::load({"x"}, {"fetch_name_0"}, mConfig.recModel.c_str(), mRuntime, &mconfig));
        if (mDet == nullptr || mRec == nullptr) {
            MNN_ERROR("OCR: failed to load det/rec models\n");
            return false;
        }
        return true;
    }

    std::vector<OcrResult> run(const std::string& imagePath) {
        std::vector<OcrResult> results;
        VARP bgr = CV::imread(imagePath);
        if (bgr == nullptr) {
            MNN_ERROR("OCR: failed to read image %s\n", imagePath.c_str());
            return results;
        }
        auto info = bgr->getInfo();
        int srcH = info->dim[0], srcW = info->dim[1];
        VARP rgb = CV::cvtColor(bgr, CV::COLOR_BGR2RGB);

        // ---------------- Detection ----------------
        int limit = mConfig.detLimitSideLen;
        float ratio = 1.f;
        if (std::max(srcH, srcW) > limit) {
            ratio = static_cast<float>(limit) / std::max(srcH, srcW);
        }
        int detH = std::max(32, static_cast<int>(std::round(srcH * ratio / 32.f)) * 32);
        int detW = std::max(32, static_cast<int>(std::round(srcW * ratio / 32.f)) * 32);
        VARP detIn = CV::resize(rgb, {detW, detH}, 0, 0, CV::INTER_LINEAR, -1, kDetMean, kDetNorm);
        detIn = _Convert(_Unsqueeze(detIn, {0}), NCHW);  // [1,3,detH,detW]

        VARP detOut = mDet->onForward({detIn})[0];
        detOut = _Convert(detOut, NCHW);
        const float* prob = detOut->readMap<float>();  // [1,1,detH,detW]

        float scaleX = static_cast<float>(srcW) / detW;
        float scaleY = static_cast<float>(srcH) / detH;
        auto boxes = detectBoxes(prob, detH, detW, scaleX, scaleY, srcW, srcH);

        // ---------------- Recognition ----------------
        for (auto& q : boxes) {
            VARP crop = cropQuad(rgb, q);
            if (crop == nullptr) {
                continue;
            }
            std::string text;
            float score = 0.f;
            recognize(crop, text, score);
            if (text.empty()) {
                continue;
            }
            OcrResult r;
            for (int i = 0; i < 4; ++i) {
                r.box.emplace_back(static_cast<int>(q.x[i]), static_cast<int>(q.y[i]));
            }
            r.text = text;
            r.score = score;
            results.push_back(std::move(r));
        }
        return results;
    }

    std::string runToText(const std::string& imagePath) {
        auto results = run(imagePath);
        std::string out;
        for (size_t i = 0; i < results.size(); ++i) {
            out += results[i].text;
            if (i + 1 < results.size()) {
                out += "\n";
            }
        }
        return out;
    }

private:
    std::vector<Quad> detectBoxes(const float* prob, int h, int w, float scaleX, float scaleY,
                                  int srcW, int srcH) {
        // Binarize the probability map.
        std::vector<uint8_t> bin(static_cast<size_t>(h) * w);
        for (size_t i = 0; i < bin.size(); ++i) {
            bin[i] = prob[i] > mConfig.detDbThresh ? 255 : 0;
        }
        VARP binVar = _Const(bin.data(), {h, w, 1}, NHWC, halide_type_of<uint8_t>());
        auto contours = CV::findContours(binVar, CV::RETR_LIST, CV::CHAIN_APPROX_SIMPLE);

        std::vector<Quad> quads;
        for (auto& c : contours) {
            auto cinfo = c->getInfo();
            int npts = cinfo->dim[0];
            if (npts < 4) {
                continue;
            }
            CV::RotatedRect rect = CV::minAreaRect(c);
            float minSide = std::min(rect.size.width, rect.size.height);
            if (minSide < 3.f) {
                continue;
            }
            VARP bp = CV::boxPoints(rect);
            const float* pts = bp->readMap<float>();  // [4,2]
            Quad q;
            for (int i = 0; i < 4; ++i) {
                q.x[i] = pts[i * 2];
                q.y[i] = pts[i * 2 + 1];
            }
            // Mean probability inside the box -> filter weak detections.
            if (boxScore(prob, h, w, q) < mConfig.detDbBoxThresh) {
                continue;
            }
            q = unclip(q, mConfig.detDbUnclipRatio);
            // Re-fit a tight rectangle around the expanded quad.
            std::vector<int> pi(8);
            for (int i = 0; i < 4; ++i) {
                pi[i * 2] = static_cast<int>(std::round(q.x[i]));
                pi[i * 2 + 1] = static_cast<int>(std::round(q.y[i]));
            }
            VARP piVar = _Const(pi.data(), {4, 1, 2}, NHWC, halide_type_of<int>());
            CV::RotatedRect rect2 = CV::minAreaRect(piVar);
            VARP bp2 = CV::boxPoints(rect2);
            const float* pts2 = bp2->readMap<float>();
            for (int i = 0; i < 4; ++i) {
                q.x[i] = clampf(pts2[i * 2] * scaleX, 0.f, static_cast<float>(srcW - 1));
                q.y[i] = clampf(pts2[i * 2 + 1] * scaleY, 0.f, static_cast<float>(srcH - 1));
            }
            quads.push_back(orderPoints(q));
        }
        // Sort top-to-bottom, then left-to-right.
        std::sort(quads.begin(), quads.end(), [](const Quad& a, const Quad& b) {
            float ay = std::min(std::min(a.y[0], a.y[1]), std::min(a.y[2], a.y[3]));
            float by = std::min(std::min(b.y[0], b.y[1]), std::min(b.y[2], b.y[3]));
            if (std::fabs(ay - by) > 10.f) {
                return ay < by;
            }
            float ax = std::min(std::min(a.x[0], a.x[1]), std::min(a.x[2], a.x[3]));
            float bx = std::min(std::min(b.x[0], b.x[1]), std::min(b.x[2], b.x[3]));
            return ax < bx;
        });
        return quads;
    }

    // Mean probability over the tight axis-aligned bounding box of a quad
    // (PaddleOCR box_score_fast). For near-horizontal text lines this closely
    // approximates the polygon mean.
    float boxScore(const float* prob, int h, int w, const Quad& q) {
        float minX = q.x[0], maxX = q.x[0], minY = q.y[0], maxY = q.y[0];
        for (int i = 1; i < 4; ++i) {
            minX = std::min(minX, q.x[i]);
            maxX = std::max(maxX, q.x[i]);
            minY = std::min(minY, q.y[i]);
            maxY = std::max(maxY, q.y[i]);
        }
        int x0 = std::max(0, static_cast<int>(std::floor(minX)));
        int x1 = std::min(w - 1, static_cast<int>(std::ceil(maxX)));
        int y0 = std::max(0, static_cast<int>(std::floor(minY)));
        int y1 = std::min(h - 1, static_cast<int>(std::ceil(maxY)));
        if (x1 <= x0 || y1 <= y0) {
            return 0.f;
        }
        double sum = 0.0;
        int cnt = 0;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                sum += prob[y * w + x];
                ++cnt;
            }
        }
        return cnt > 0 ? static_cast<float>(sum / cnt) : 0.f;
    }

    VARP cropQuad(VARP rgb, const Quad& q) {
        int wTop = static_cast<int>(std::round(std::hypot(q.x[0] - q.x[1], q.y[0] - q.y[1])));
        int wBot = static_cast<int>(std::round(std::hypot(q.x[3] - q.x[2], q.y[3] - q.y[2])));
        int hLft = static_cast<int>(std::round(std::hypot(q.x[0] - q.x[3], q.y[0] - q.y[3])));
        int hRgt = static_cast<int>(std::round(std::hypot(q.x[1] - q.x[2], q.y[1] - q.y[2])));
        int cw = std::max(std::max(wTop, wBot), 1);
        int ch = std::max(std::max(hLft, hRgt), 1);

        CV::Point src[4], dst[4];
        for (int i = 0; i < 4; ++i) {
            src[i].set(q.x[i], q.y[i]);
        }
        dst[0].set(0.f, 0.f);
        dst[1].set(static_cast<float>(cw), 0.f);
        dst[2].set(static_cast<float>(cw), static_cast<float>(ch));
        dst[3].set(0.f, static_cast<float>(ch));
        CV::Matrix M = CV::getPerspectiveTransform(src, dst);
        VARP crop = CV::warpPerspective(rgb, M, {cw, ch});
        // Rotate tall crops 90° CCW (matches np.rot90) so text reads left-to-right.
        if (static_cast<float>(ch) / cw >= 1.5f) {
            crop = _Reverse(_Transpose(crop, {1, 0, 2}), _Scalar<int>(0));
        }
        return crop;
    }

    void recognize(VARP crop, std::string& text, float& score) {
        auto info = crop->getInfo();
        int ch = info->dim[0], cw = info->dim[1];
        int recH = mConfig.recImgHeight;
        int recW = std::max(1, static_cast<int>(std::ceil(recH * static_cast<float>(cw) / ch)));
        VARP recIn = CV::resize(crop, {recW, recH}, 0, 0, CV::INTER_LINEAR, -1, kRecMean, kRecNorm);
        recIn = _Convert(_Unsqueeze(recIn, {0}), NCHW);  // [1,3,recH,recW]

        VARP out = mRec->onForward({recIn})[0];
        out = _Convert(out, NCHW);
        auto oinfo = out->getInfo();
        int T = oinfo->dim[1];
        int V = oinfo->dim[2];
        const float* data = out->readMap<float>();  // [1,T,V]

        // CTC greedy decode: argmax per step, drop blank (index 0), collapse repeats.
        std::string result;
        double scoreSum = 0.0;
        int scoreCnt = 0;
        int last = -1;
        for (int t = 0; t < T; ++t) {
            const float* row = data + static_cast<size_t>(t) * V;
            int best = 0;
            float bestP = row[0];
            for (int v = 1; v < V; ++v) {
                if (row[v] > bestP) {
                    bestP = row[v];
                    best = v;
                }
            }
            if (best != 0 && best != last) {
                if (best < static_cast<int>(mKeys.size())) {
                    result += mKeys[best];
                    scoreSum += bestP;
                    ++scoreCnt;
                }
            }
            last = best;
        }
        text = result;
        score = scoreCnt > 0 ? static_cast<float>(scoreSum / scoreCnt) : 0.f;
    }

    OcrConfig mConfig;
    std::vector<std::string> mKeys;
    std::shared_ptr<Executor::RuntimeManager> mRuntime;
    std::shared_ptr<Module> mDet;
    std::shared_ptr<Module> mRec;
};

Ocr::Ocr(const OcrConfig& config) : mImpl(new Impl(config)) {}
Ocr::~Ocr() = default;
bool Ocr::load() { return mImpl->load(); }
std::vector<OcrResult> Ocr::run(const std::string& imagePath) { return mImpl->run(imagePath); }
std::string Ocr::runToText(const std::string& imagePath) { return mImpl->runToText(imagePath); }

}  // namespace OCR
}  // namespace MNN
