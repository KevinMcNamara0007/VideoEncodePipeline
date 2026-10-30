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
/// The preview is the survivor keyframes, one after another, on the SOURCE
/// clock: each survivor stays on screen until the next survivor's timestamp
/// (a dropped near-duplicate leaves its stand-in up for its slot), the last
/// one for --hold seconds. Preview second t is source millisecond 1000 t, so
/// a fact's `point_in_time` seeks the preview without a lookup. Boxes are in
/// encoder-frame pixels and are scaled by src_w/frame_w and src_h/frame_h,
/// matching ffmpeg scale=W:H (stretched, not letterboxed).
///
/// Each source is decoded once, through the same FFmpegFramePipe and fps
/// filter the encoder sampled it with (config.json target_fps), at source
/// size: survivor frame_idx N is frame N of that stream, exactly the frame
/// the boxes were found on. (One seek per survivor decoded from the start
/// each time: 326 survivors took 3 min 52 s. One pass takes seconds.)
///
/// Beside the preview it writes `<preview>.index.json`, the lookup the
/// appliance's page reads to play the frames a fact names: every survivor
/// in preview order with its source time, preview time, scene and boxes
/// (object id, class, words), and the deduplicated objects (objects.jsonl)
/// with the frames each one was seen on.
///
///     aerial_paint <out_dir> <preview.mp4> [--hold 0.5]
#include "ehc/media/ffmpeg_pipe.hpp"
#include "ehc/util/subprocess.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

/// The rate the encoder sampled at: config.json target_fps, else the ratio
/// frame_idx / ts_ms of the first row that has one, else 2.
double sample_fps(const fs::path& out_dir, const std::vector<json>& rows) {
    std::ifstream in(out_dir / "config.json");
    if (in) {
        try {
            const json cfg = json::parse(in);
            if (cfg.contains("target_fps") && cfg["target_fps"].is_number() && cfg["target_fps"].get<double>() > 0)
                return cfg["target_fps"].get<double>();
        } catch (const std::exception&) {
        }
    }
    for (const auto& r : rows) {
        const double idx = r.value("frame_idx", 0.0), ts = r.value("ts_ms", 0.0);
        if (idx > 0 && ts > 0) return idx * 1000.0 / ts;
    }
    return 2.0;
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
        // fact:false (the encoder withheld the object: one look under the
        // floor) is drawn grey and thin, so the preview shows what the lake
        // was not told.
        const bool fact = !(b.contains("fact") && b["fact"].is_boolean() && !b["fact"].get<bool>());
        const cv::Scalar col = fact ? color_for(label) : cv::Scalar{128, 128, 128};
        cv::rectangle(img, {x0, y0}, {x1, y1}, col, fact ? 2 : 1);
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
        if (!fact) text += "  withheld";
        cv::putText(img, text, {x0, std::max(20, y0 - 8)}, cv::FONT_HERSHEY_SIMPLEX, 0.6, col, fact ? 2 : 1,
                    cv::LINE_AA);
    }
}

[[noreturn]] void die(const std::string& msg) {
    std::cerr << msg << "\n";
    std::exit(1);
}

/// One survivor of the preview: its row, and where it sits on the preview clock.
struct Cue {
    size_t row;          // index into rows
    double t;            // preview seconds this survivor appears
    double duration;     // seconds it stays up
};

/// The lookup beside the preview (see the file comment). `objects` is
/// objects.jsonl when the encode wrote one.
json preview_index(const fs::path& out_dir, const fs::path& preview, double fps, const std::vector<json>& rows,
                   const std::vector<Cue>& cues) {
    json videos = json::array(), frames = json::array();
    std::map<int, size_t> video_slot;
    for (const auto& c : cues) {
        const json& r = rows[c.row];
        const int vid = r.value("video_id", 0);
        if (!video_slot.count(vid)) {
            video_slot[vid] = videos.size();
            videos.push_back({{"video_id", vid}, {"source_path", r.value("source_path", "")},
                              {"src_w", r.value("src_w", 0)}, {"src_h", r.value("src_h", 0)},
                              {"frame_w", r.value("frame_w", 0)}, {"frame_h", r.value("frame_h", 0)},
                              {"first_t", c.t}, {"last_t", c.t + c.duration}});
        } else {
            videos[video_slot[vid]]["last_t"] = c.t + c.duration;
        }
        json boxes = json::array();
        for (const auto& b : r.value("boxes", json::array())) {
            if (b.contains("overlay") && b["overlay"].is_boolean() && !b["overlay"].get<bool>()) continue;
            boxes.push_back({{"object_id", b.value("object_id", -1)}, {"class", b.value("class", "")},
                             {"ocr", b.value("ocr", json::array())}, {"confidence", b.value("confidence", 0.0)},
                             {"fact", b.value("fact", true)}, {"xyxy", b.value("xyxy", json::array())}});
        }
        frames.push_back({{"i", frames.size()}, {"video_id", vid}, {"frame_idx", r.value("frame_idx", 0)},
                          {"ts_ms", r.value("ts_ms", 0.0)}, {"t", c.t}, {"end", c.t + c.duration},
                          {"scene", r.value("scene", "")}, {"boxes", boxes}});
    }
    json objects = json::array();
    if (std::ifstream obj(out_dir / "objects.jsonl"); obj) {
        for (std::string line; std::getline(obj, line);) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            const json o = json::parse(line);
            json seen = json::array();
            for (const auto& f : o.value("frames", json::array())) seen.push_back(f.value("frame_idx", 0));
            objects.push_back({{"object_id", o.value("object_id", -1)}, {"video_id", o.value("video_id", 0)},
                               {"class", o.value("class", "")}, {"source", o.value("source", "")},
                               {"confidence", o.value("confidence", 0.0)}, {"ocr", o.value("ocr", json::array())},
                               {"fact", o.value("fact", true)},
                               {"host_id", o.value("host_id", json())}, {"first_ts_ms", o.value("first_ts_ms", 0.0)},
                               {"last_ts_ms", o.value("last_ts_ms", 0.0)}, {"frames", seen}});
        }
    }
    return json{{"version", 1},
                {"preview", preview.filename().string()},
                {"clock", "source"},
                {"sample_fps", fps},
                {"videos", videos},
                {"frames", frames},
                {"objects", objects}};
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    double hold = 0.5;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "-h" || k == "--help") {
            std::cout << "usage: aerial_paint <out_dir> <preview.mp4> [--hold SECONDS]\n"
                         "  Survivors play on the source clock (each until the next survivor's\n"
                         "  timestamp); writes <preview>.index.json beside the preview.\n"
                         "  --hold  seconds the LAST survivor of each source stays on screen (default 0.5)\n";
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
    const double fps = sample_fps(out_dir, rows);

    // Rows by source, in file order; each source is decoded once.
    std::vector<std::string> sources;
    std::map<std::string, std::map<int, std::vector<size_t>>> wanted;   // source -> frame_idx -> rows
    for (size_t i = 0; i < rows.size(); ++i) {
        const std::string src = rows[i].value("source_path", "");
        if (!wanted.count(src)) sources.push_back(src);
        wanted[src][rows[i].value("frame_idx", 0)].push_back(i);
    }
    std::vector<fs::path> painted(rows.size());
    size_t n_painted = 0;
    for (const auto& src : sources) {
        auto& by_idx = wanted[src];
        const json& first = rows[by_idx.begin()->second.front()];
        const int src_w = first.value("src_w", 0), src_h = first.value("src_h", 0);
        if (src.empty() || src_w <= 0 || src_h <= 0) {
            std::cerr << "[paint] skip " << by_idx.size() << " row(s): no source size for '" << src << "'\n";
            continue;
        }
        const int last = by_idx.rbegin()->first;
        ehc::media::FFmpegFramePipe pipe(src, fps, src_w, src_h, ehc::media::ColorMode::rgb);
        while (auto f = pipe.next()) {
            if (f->frame_idx > last) break;
            const auto it = by_idx.find(f->frame_idx);
            if (it == by_idx.end()) continue;
            cv::Mat img;
            cv::cvtColor(cv::Mat(f->h, f->w, CV_8UC3, const_cast<uint8_t*>(f->data)), img, cv::COLOR_RGB2BGR);
            for (size_t i : it->second) {
                cv::Mat plate = it->second.size() > 1 ? img.clone() : img;
                draw_row(plate, rows[i]);
                // yuv420p wants even sizes
                if (plate.cols % 2 || plate.rows % 2)
                    cv::copyMakeBorder(plate, plate, 0, plate.rows % 2, 0, plate.cols % 2, cv::BORDER_REPLICATE);
                char name[32];
                std::snprintf(name, sizeof name, "frame_%05zu.png", i);
                painted[i] = plates / name;
                cv::imwrite(painted[i].string(), plate);
                ++n_painted;
            }
            by_idx.erase(it);
        }
        if (!by_idx.empty())
            std::cerr << "[paint] " << src << ": " << by_idx.size() << " survivor frame(s) not in the stream"
                      << (ehc::util::Subprocess::status_failed(pipe.exit_status()) ? " (ffmpeg failed)" : "") << "\n";
    }
    if (n_painted == 0) die("no frames painted");

    // The preview clock: a survivor stays up until the next survivor of the
    // same source (its own slot plus those of the near-duplicates it stands
    // for); the last of a source, and a lone one, for --hold.
    std::vector<Cue> cues;
    double t = 0.0;
    for (size_t i = 0; i < painted.size(); ++i) {
        if (painted[i].empty()) continue;
        size_t next = i + 1;
        while (next < painted.size() && painted[next].empty()) ++next;
        double d = hold;
        if (next < painted.size() && rows[next].value("source_path", "") == rows[i].value("source_path", "")) {
            const double gap = (rows[next].value("ts_ms", 0.0) - rows[i].value("ts_ms", 0.0)) / 1000.0;
            if (gap > 0) d = gap;
        }
        cues.push_back({i, t, d});
        t += d;
    }

    // One image per sample period, numbered in preview order, read by the
    // image2 demuxer at -framerate fps: preview frame n sits at exactly n/fps,
    // the clock the index describes. Every cue is a whole number of periods
    // (survivors sit on the sample grid; a dropped frame's slot is one
    // period), so a held survivor fills each of its periods; the slot files
    // are links to the painted plates, not copies.
    //
    // The concat demuxer with `duration` entries, even re-clocked with
    // -r/-fps_mode cfr, is NOT this clock: at the first held cue (a 1 s entry)
    // it showed the next survivor one period early, and every later frame
    // played 0.5 s ahead of its index entry -- 305 of 338 preview frames off
    // by one on drone_raw, the player seeking to frame N and showing N+1
    // (measured 2026-10-30 by matching each preview frame to its plate).
    const fs::path slots = out_dir / "preview_slots";
    {
        std::error_code ec;
        fs::remove_all(slots, ec);  // a shorter run must not inherit stale slots
        fs::create_directories(slots);
        size_t n = 0;
        for (const auto& c : cues) {
            const long k = std::max(1L, std::lround(c.duration * fps));
            for (long j = 0; j < k; ++j) {
                char name[32];
                std::snprintf(name, sizeof name, "slot_%05zu.png", n++);
                fs::create_symlink(fs::absolute(painted[c.row]), slots / name, ec);
                if (ec) die("cannot link " + (slots / name).string() + ": " + ec.message());
            }
        }
    }
    if (preview.has_parent_path()) fs::create_directories(preview.parent_path());
    try {
        char rate[32];
        std::snprintf(rate, sizeof rate, "%.6g", fps);
        (void)ehc::util::Subprocess::capture({"ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-framerate", rate,
                                              "-i", (slots / "slot_%05d.png").string(),
                                              "-video_track_timescale", "1000", "-pix_fmt", "yuv420p",
                                              // moov first: a page seeking over HTTP Range reads the index
                                              // from the head of the file, not after a round trip to the tail.
                                              "-movflags", "+faststart", preview.string()});
    } catch (const std::exception& e) {
        die(std::string("ffmpeg failed: ") + e.what());
    }
    fs::path index = preview;
    index.replace_extension(".index.json");
    {
        std::ofstream fh(index);
        if (!fh) die("cannot write " + index.string());
        fh << preview_index(out_dir, preview, fps, rows, cues).dump() << "\n";
    }
    std::cout << "[paint] " << cues.size() << " frames, " << t << " s -> " << preview.string() << "  index "
              << index.string() << std::endl;
    return 0;
}
