// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
/// ocr_bench <reader> <views-dir> -- every PNG in the dir (the crops
/// aerial_encode keeps under EWORLD_KEEP_VIEWS) through one reader; one
/// JSON line per view {view, seconds, words}, then a summary line on stderr.
///
///   reader = tess:<psm>[:zoom]   eworld::Tesseract on the crop, enlarged
///                                `zoom` times first (default 1)
///          = door               the HaloCore door as the encode uses it;
///                                HALOCORE_FIGURES_TASKS / _THREADS /
///                                HALOCORE_DOOR_LIB apply. One call for the
///                                batch, so seconds is the batch mean.
#include "florence.hpp"
#include "ocr.hpp"
#include "rules.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: ocr_bench <tess:<psm>[:zoom]|door> <views-dir>\n";
        return 2;
    }
    const std::string reader = argv[1];
    std::vector<fs::path> views;
    for (const auto& e : fs::directory_iterator(argv[2]))
        if (e.is_regular_file() && e.path().extension() == ".png") views.push_back(e.path());
    std::sort(views.begin(), views.end());
    if (views.empty()) {
        std::cerr << "no PNG views under " << argv[2] << "\n";
        return 1;
    }
    std::map<std::string, std::vector<std::string>> words;
    std::map<std::string, double> seconds;
    const double t0 = now_s();
    if (reader.rfind("tess:", 0) == 0) {
        const std::string spec = reader.substr(5);
        const auto colon = spec.find(':');
        const int psm = std::stoi(spec.substr(0, colon));
        const double zoom = colon == std::string::npos ? 1.0 : std::stod(spec.substr(colon + 1));
        eworld::Tesseract tess;
        for (const auto& p : views) {
            cv::Mat img = cv::imread(p.string(), cv::IMREAD_COLOR);
            if (zoom != 1.0) cv::resize(img, img, cv::Size(), zoom, zoom, cv::INTER_CUBIC);
            const double t = now_s();
            std::vector<std::string> ws;
            for (const auto& w : tess.words(img, psm)) {
                const auto tok = eworld::ocr_token(w.text);
                if (tok && eworld::keep_stencil(tok->substr(4)) &&
                    std::find(ws.begin(), ws.end(), tok->substr(4)) == ws.end())
                    ws.push_back(tok->substr(4));
            }
            seconds[p.stem().string()] = now_s() - t;
            words[p.stem().string()] = ws;
        }
    } else if (reader == "door") {
        eworld::HaloFlorence door;
        std::vector<std::string> paths;
        for (const auto& p : views) paths.push_back(p.string());
        const double t = now_s();
        const auto texts = door.read_many(paths);
        const double per = (now_s() - t) / static_cast<double>(views.size());
        for (const auto& p : views) {
            const std::string stem = p.stem().string();
            seconds[stem] = per;
            const auto it = texts.find(stem);
            words[stem] = it == texts.end() ? std::vector<std::string>{} : eworld::figure_words(it->second);
        }
    } else {
        std::cerr << "unknown reader " << reader << "\n";
        return 2;
    }
    size_t with_text = 0;
    for (const auto& p : views) {
        const std::string stem = p.stem().string();
        with_text += !words[stem].empty();
        std::cout << nlohmann::json{{"view", stem}, {"seconds", seconds[stem]}, {"words", words[stem]}}.dump()
                  << "\n";
    }
    std::cerr << reader << ": " << views.size() << " views, " << with_text << " with text, "
              << (now_s() - t0) << " s total\n";
    return 0;
}
