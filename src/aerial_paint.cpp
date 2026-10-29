// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
/// aerial_paint -- paint overlay.jsonl back onto the source frames and write
/// a preview mp4.
///
/// The preview is the survivor keyframes, one after another, each held for
/// --hold seconds (0.5 = 2 fps). Boxes are in encoder-frame pixels and are
/// scaled by src_w/frame_w and src_h/frame_h, matching ffmpeg scale=W:H
/// (stretched, not letterboxed).
///
///     aerial_paint <out_dir> <preview.mp4> [--hold 0.5]
#include "ehc/util/subprocess.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string input_token(const std::string& path) {
    std::error_code ec;
    return fs::exists(path, ec) ? "file:" + path : path;
}

/// One frame at ts_ms. Seek after -i so the sample clock stays close.
cv::Mat grab_frame(const std::string& source, double ts_ms) {
    char sec[32];
    std::snprintf(sec, sizeof sec, "%.3f", std::max(0.0, ts_ms / 1000.0));
    std::string png;
    try {
        png = ehc::util::Subprocess::capture({"ffmpeg", "-nostdin", "-loglevel", "error", "-i", input_token(source),
                                              "-ss", sec, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png",
                                              "-"});
    } catch (const std::exception&) {
        return {};
    }
    if (png.empty()) return {};
    const cv::Mat buf(1, static_cast<int>(png.size()), CV_8UC1, png.data());
    return cv::imdecode(buf, cv::IMREAD_COLOR);
}

/// A stable colour per label (FNV-1a), so a class keeps its colour across runs.
cv::Scalar color_for(const std::string& label) {
    static const cv::Scalar palette[6] = {
        {60, 220, 80}, {40, 180, 255}, {80, 80, 255}, {200, 180, 40}, {220, 80, 220}, {180, 180, 180},
    };
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : label) h = (h ^ c) * 1099511628211ULL;
    return palette[h % 6];
}

void draw_row(cv::Mat& img, const json& row) {
    const int src_w = img.cols, src_h = img.rows;
    auto num_or = [&](const char* key, double fallback) {
        const auto it = row.find(key);
        return it != row.end() && it->is_number() && it->get<double>() != 0 ? it->get<double>() : fallback;
    };
    const double fw = num_or("frame_w", src_w), fh = num_or("frame_h", src_h);
    const double sx = fw ? src_w / fw : 1.0, sy = fh ? src_h / fh : 1.0;
    const std::string scene = row.value("scene", "");
    if (!scene.empty()) {
        cv::rectangle(img, {0, 0}, {src_w, 36}, {0, 0, 0}, cv::FILLED);
        cv::putText(img, scene, {12, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {240, 240, 240}, 2, cv::LINE_AA);
    }
    for (const auto& b : row.value("boxes", json::array())) {
        if (b.contains("overlay") && b["overlay"].is_boolean() && !b["overlay"].get<bool>()) continue;
        std::string label = b.value("class", "");
        if (label.empty()) label = "?";
        const json xy = b.contains("xyxy") ? b["xyxy"] : json::array({0, 0, 0, 0});
        const int x0 = static_cast<int>(std::lround(xy[0].get<double>() * sx));
        const int y0 = static_cast<int>(std::lround(xy[1].get<double>() * sy));
        const int x1 = static_cast<int>(std::lround(xy[2].get<double>() * sx));
        const int y1 = static_cast<int>(std::lround(xy[3].get<double>() * sy));
        const cv::Scalar col = color_for(label);
        cv::rectangle(img, {x0, y0}, {x1, y1}, col, 2);
        const json ocr = b.value("ocr", json::array());
        std::string text = label;
        if (label.rfind("ocr:", 0) == 0) text = label.substr(4);
        else {
            // "#id" is the object in objects.jsonl: the same id across frames is one object.
            if (b.contains("object_id") && b["object_id"].is_number_integer() && b["object_id"].get<long>() >= 0)
                text += " #" + std::to_string(b["object_id"].get<long>());
            // Up to three of the object's words (a sign is a few words; a plate one).
            for (size_t k = 0; k < std::min<size_t>(ocr.size(), 3); ++k)
                text += (k ? " " : "  ") + ocr[k].get<std::string>();
        }
        if (b.contains("confidence") && b["confidence"].is_number_float()) {
            char c[16];
            std::snprintf(c, sizeof c, "  %.2f", b["confidence"].get<double>());
            text += c;
        }
        cv::putText(img, text, {x0, std::max(20, y0 - 8)}, cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv::LINE_AA);
    }
}

[[noreturn]] void die(const std::string& msg) {
    std::cerr << msg << "\n";
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    double hold = 0.5;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "-h" || k == "--help") {
            std::cout << "usage: aerial_paint <out_dir> <preview.mp4> [--hold SECONDS]\n"
                         "  --hold  seconds each survivor stays on screen (default 0.5)\n";
            return 0;
        }
        if (k == "--hold" && i + 1 < argc) hold = std::stod(argv[++i]);
        else pos.push_back(k);
    }
    if (pos.size() != 2) die("usage: aerial_paint <out_dir> <preview.mp4> [--hold SECONDS]");
    const fs::path out_dir = pos[0], preview = pos[1];

    const fs::path overlay_path = out_dir / "overlay.jsonl";
    std::ifstream in(overlay_path);
    if (!in) die("missing " + overlay_path.string());
    std::vector<json> rows;
    for (std::string line; std::getline(in, line);)
        if (line.find_first_not_of(" \t\r\n") != std::string::npos) rows.push_back(json::parse(line));
    if (rows.empty()) die("overlay.jsonl is empty");

    const fs::path plates = out_dir / "preview_frames";
    fs::create_directories(plates);
    std::vector<fs::path> written;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        const double ts = row.contains("ts_ms") && row["ts_ms"].is_number() ? row["ts_ms"].get<double>() : 0.0;
        cv::Mat img = grab_frame(row.value("source_path", ""), ts);
        if (img.empty()) {
            std::cerr << "[paint] skip row " << i << ": no frame from " << row.value("source_path", "") << "\n";
            continue;
        }
        draw_row(img, row);
        // yuv420p wants even sizes
        if (img.cols % 2 || img.rows % 2)
            cv::copyMakeBorder(img, img, 0, img.rows % 2, 0, img.cols % 2, cv::BORDER_REPLICATE);
        char name[32];
        std::snprintf(name, sizeof name, "frame_%05zu.png", i);
        const fs::path path = plates / name;
        cv::imwrite(path.string(), img);
        written.push_back(path);
    }
    if (written.empty()) die("no frames painted");

    // Concat resolves relative paths from the list file's directory.
    const fs::path lst = out_dir / "preview_frames.txt";
    {
        std::ofstream fh(lst);
        for (const auto& p : written) fh << "file '" << fs::absolute(p).string() << "'\nduration " << hold << "\n";
        fh << "file '" << fs::absolute(written.back()).string() << "'\n";
    }
    if (preview.has_parent_path()) fs::create_directories(preview.parent_path());
    try {
        (void)ehc::util::Subprocess::capture({"ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-f", "concat",
                                              "-safe", "0", "-i", lst.string(), "-fps_mode", "vfr", "-pix_fmt",
                                              "yuv420p", preview.string()});
    } catch (const std::exception& e) {
        die(std::string("ffmpeg failed: ") + e.what());
    }
    std::cout << "[paint] " << written.size() << " frames -> " << preview.string() << std::endl;
    return 0;
}
