// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
/// aerial_encode -- encode aerial clips into the EHC streaming video encoder
/// with object labels and per-object text fused into each keyframe vector,
/// and write the boxes and the object index the vector does not store. See
/// recipe_aerial_overlay.md.
///
///     aerial_encode <clips_dir> <out_dir> --fps 2 --frame 640x640 --ocr florence
#include "aerial_detector.hpp"

#include "ehc/index/bsc_lsh_index.hpp"
#include "ehc/media/ffmpeg_pipe.hpp"
#include "ehc/media/streaming_video_encoder.hpp"
#include "ehc/util/subprocess.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef EWORLD_SOURCE_DIR
#define EWORLD_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
using namespace eworld;

namespace {

constexpr int      kDim        = 16384;
constexpr int      kK          = 128;
constexpr int      kNumTables  = 8;
constexpr int      kHashSize   = 16;
constexpr uint64_t kSeed       = 2300;
constexpr uint64_t kTokenSeed  = 2400;   // StreamingVideoConfig::detector_token_seed default

const char* kUsage =
    "usage: aerial_encode <source_dir> <out_dir> [options]\n"
    "  --limit N            encode the first N clips (0 = all)\n"
    "  --exts LIST          default mp4,webm,mov,mkv,avi,m4v\n"
    "  --fps F              sample rate (default 2)\n"
    "  --frame WxH          encoder / YOLO / OCR frame (default 640x640, min 320x320)\n"
    "  --objects PATH       object layer config (default config/object_layer.json)\n"
    "  --model PATH         override the config's COCO head .onnx\n"
    "  --world-model PATH   override the config's YOLO-World head .onnx\n"
    "  --conf C             override the config's COCO confidence\n"
    "  --ocr MODE           florence (default; HaloCore door) | tesseract | none.\n"
    "                       A few reads per deduplicated object, spread over its frames,\n"
    "                       on a reader thread while the object layer runs on.\n"
    "  --threads N          object-layer workers, one frame each (default 2 on 8+ cores, else 1)\n";

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string input_token(const std::string& path) {
    std::error_code ec;
    return fs::exists(path, ec) ? "file:" + path : path;
}

json video_meta(const std::string& src) {
    const json probe = json::parse(ehc::util::Subprocess::capture(
        {"ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", input_token(src)}));
    double fps = 0.0, duration_ms = 0.0;
    std::string codec, container;
    int total_frames = 0, src_w = 0, src_h = 0;
    auto num = [](const json& v) -> double {
        if (v.is_number()) return v.get<double>();
        if (v.is_string()) try { return std::stod(v.get<std::string>()); } catch (...) {}
        return 0.0;
    };
    for (const auto& s : probe.value("streams", json::array())) {
        if (s.value("codec_type", "") != "video") continue;
        codec = s.value("codec_name", "");
        src_w = static_cast<int>(num(s.value("width", json())));
        src_h = static_cast<int>(num(s.value("height", json())));
        for (const char* key : {"avg_frame_rate", "r_frame_rate"}) {
            const std::string val = s.value(key, "0/0");
            const auto slash = val.find('/');
            if (slash == std::string::npos) continue;
            try {
                const double n = std::stod(val.substr(0, slash)), d = std::stod(val.substr(slash + 1));
                if (d > 0 && n > 0) { fps = n / d; break; }
            } catch (...) {}
        }
        total_frames = static_cast<int>(num(s.value("nb_frames", json())));
        break;
    }
    const json fmt = probe.value("format", json::object());
    container = fmt.value("format_name", "");
    duration_ms = num(fmt.value("duration", json())) * 1000.0;
    return json{{"source_path", src}, {"fps", fps > 0 ? fps : 30.0}, {"duration_ms", duration_ms},
                {"total_frames", total_frames}, {"codec", codec}, {"container", container},
                {"src_w", src_w}, {"src_h", src_h}};
}

/// `fact`: whether the box's object is corroborated (ObjectLayerConfig::
/// corroborated) and so emitted as triples. The paint draws fact:false grey.
json box_json(const Box& b, bool fact) {
    return json{{"object_id", b.object_id}, {"class", b.cls}, {"detected", b.detected}, {"confidence", b.confidence},
                {"xyxy", {b.xyxy[0], b.xyxy[1], b.xyxy[2], b.xyxy[3]}}, {"ocr", b.ocr}, {"source", b.source},
                {"fact", fact}};
}

/// The clip as triples for the HaloCore ingest front door
/// (`halocore ingest --source out/triples.jsonl`, or POST /v1/ingest).
///
/// Subjects are names, not ids ("frame 26", "truck", "drone raw"): a colon
/// id makes a store record-keyed, and a record-keyed store gets no
/// surfaces. Relations are slugs the relation-surface lexicon already has
/// forms for ("part of", "point in time", "shown in", "story setting",
/// "on-screen text", "title", "also known as"). Every sampled frame is a
/// subject: a frame the encoder dropped as a near-duplicate is
/// said_to_be_the_same_as the survivor that stands for it. Only a box
/// whose object is in `facts` (corroborated: seen on enough frames, or
/// once confidently) becomes a depicts / title fact; the heads' low
/// thresholds are for tracking, and the lake scores every triple alike, so
/// one weak look must not become the first answer.
void write_triples(std::ostream& out, const fs::path& src, const json& vmeta,
                   const std::vector<ehc::media::EncodedFrame>& encoded, const std::vector<FrameRecord>& records,
                   int n_input, double fps, const ObjectLayerConfig& layer, const std::set<long>& facts) {
    auto emit = [&](const std::string& s, const char* r, const std::string& o, const std::string& q = "") {
        out << json{{"subject", s}, {"relation", r}, {"object", o}, {"qualifier", q}}.dump() << "\n";
    };
    auto frame = [](int idx) { return "frame " + std::to_string(idx); };
    auto ms_of = [&](int idx) { return std::to_string(static_cast<long long>(std::llround(idx * 1000.0 / fps))); };
    auto name = [](std::string s) {   // "scene:air" -> "air", "weapon:rifle" -> "weapon rifle"
        if (s.rfind("scene:", 0) == 0) s = s.substr(6);
        std::replace(s.begin(), s.end(), ':', ' ');
        std::replace(s.begin(), s.end(), '_', ' ');
        return s;
    };
    const std::string video = name(src.stem().string());
    auto model = [](const std::string& p) { return p.empty() ? "" : fs::path(p).stem().string(); };
    std::string heads = model(layer.coco_model);
    if (!layer.world_model.empty()) heads += "+" + model(layer.world_model);
    emit(video, "title", video);
    emit(video, "described_by_source", heads + "+florence-2");
    emit(video, "duration", std::to_string(static_cast<long long>(std::llround(vmeta.value("duration_ms", 0.0)))) + " ms");
    emit(video, "facet_of", "eworld");

    // Which survivor stands for each sampled frame: itself, a stage-2
    // cluster member, or (a stage-1 drop) the frame it was compared to,
    // the last one kept before it.
    std::map<int, int> stands_for;
    for (const auto& ef : encoded) {
        stands_for[ef.meta.frame_idx] = ef.meta.frame_idx;
        for (const auto& m : ef.meta.also_seen_at) stands_for[m.frame_idx] = ef.meta.frame_idx;
    }
    for (int idx = 0; idx < n_input; ++idx)
        if (!stands_for.count(idx) && stands_for.count(idx - 1)) stands_for[idx] = stands_for[idx - 1];
    for (int idx = 0; idx < n_input; ++idx) {
        const std::string ms = ms_of(idx);
        emit(frame(idx), "part_of", video, ms);
        emit(frame(idx), "point_in_time", ms);
        const auto it = stands_for.find(idx);
        if (it != stands_for.end() && it->second != idx) emit(frame(idx), "said_to_be_the_same_as", frame(it->second), ms);
    }
    for (size_t i = 0; i < encoded.size() && i < records.size(); ++i) {
        const int idx = encoded[i].meta.frame_idx;
        const std::string f = frame(idx), ms = ms_of(idx);
        const auto& rec = records[i];
        if (!rec.scene.empty()) {
            emit(f, "narrative_location", name(rec.scene), ms);
            emit(name(rec.scene), "location", f, ms);
        }
        std::vector<std::string> classes, words;
        size_t n_fact = 0;
        for (const auto& b : rec.boxes) {
            if (!facts.count(b.object_id)) continue;
            ++n_fact;
            if (b.cls != "text" && std::find(classes.begin(), classes.end(), b.cls) == classes.end())
                classes.push_back(b.cls);
            for (const auto& w : b.ocr)
                if (std::find(words.begin(), words.end(), w) == words.end()) words.push_back(w);
        }
        for (const auto& c : classes) {
            emit(f, "depicts", name(c), ms);
            emit(name(c), "depicted_by", f, ms);
        }
        for (const auto& w : words) {
            emit(f, "subtitle", w, ms);
            emit(f, "title", w, ms);
            emit(w, "depicted_by", f, ms);
            emit(w, "subtitle", f, ms);
        }
        if (n_fact == 0) emit(f, "facet_of", "no-box", ms);
    }
}

struct Args {
    fs::path source_dir, out_dir;
    int limit = 0;
    std::string exts = "mp4,webm,mov,mkv,avi,m4v";
    double fps = 2.0;
    std::string frame = "640x640";
    std::string objects = (fs::path(EWORLD_SOURCE_DIR) / "config" / "object_layer.json").string();
    std::string model, world_model;
    float conf = -1.0f;
    std::string ocr = "florence";
    int threads = 0;
};

Args parse(int argc, char** argv) {
    Args a;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << k << " needs a value\n" << kUsage; std::exit(2); }
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { std::cout << kUsage; std::exit(0); }
        else if (k == "--limit") a.limit = std::stoi(val());
        else if (k == "--exts") a.exts = val();
        else if (k == "--fps") a.fps = std::stod(val());
        else if (k == "--frame") a.frame = val();
        else if (k == "--objects") a.objects = val();
        else if (k == "--model") a.model = val();
        else if (k == "--world-model") a.world_model = val();
        else if (k == "--conf") a.conf = std::stof(val());
        else if (k == "--threads") a.threads = std::stoi(val());
        else if (k == "--ocr") {
            a.ocr = val();
            if (a.ocr != "florence" && a.ocr != "tesseract" && a.ocr != "none") {
                std::cerr << "--ocr is florence, tesseract or none\n";
                std::exit(2);
            }
        }
        else if (k.rfind("--", 0) == 0) { std::cerr << "unknown option " << k << "\n" << kUsage; std::exit(2); }
        else pos.push_back(k);
    }
    if (pos.size() != 2) { std::cerr << kUsage; std::exit(2); }
    a.source_dir = pos[0];
    a.out_dir = pos[1];
    return a;
}

[[noreturn]] void die(const std::string& msg) {
    std::cerr << msg << "\n";
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse(argc, argv);

    std::string spec = args.frame;
    for (auto& c : spec) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const auto x = spec.find('x');
    if (x == std::string::npos || spec.find('x', x + 1) != std::string::npos)
        die("--frame must be WxH, for example 640x640");
    const int frame_w = std::stoi(spec.substr(0, x)), frame_h = std::stoi(spec.substr(x + 1));
    if (frame_w < 320 || frame_h < 320) die("--frame below 320x320 is too small for YOLO and OCR");

    fs::create_directories(args.out_dir);
    std::set<std::string> exts;
    {
        std::stringstream ss(args.exts);
        for (std::string e; std::getline(ss, e, ',');) {
            e.erase(0, e.find_first_not_of(" \t"));
            e.erase(e.find_last_not_of(" \t") + 1);
            for (auto& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (!e.empty()) exts.insert("." + e);
        }
    }
    std::vector<fs::path> sources;
    if (fs::is_directory(args.source_dir)) {
        for (const auto& ent : fs::recursive_directory_iterator(args.source_dir)) {
            if (!ent.is_regular_file()) continue;
            std::string ext = ent.path().extension().string();
            for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (exts.count(ext)) sources.push_back(ent.path());
        }
    }
    std::sort(sources.begin(), sources.end());   // element-wise, as Python sorts Paths
    if (args.limit > 0 && static_cast<int>(sources.size()) > args.limit) sources.resize(static_cast<size_t>(args.limit));
    if (sources.empty()) die("no clips under " + args.source_dir.string());
    std::cout << "[aerial] " << sources.size() << " clips  fps=" << args.fps << "  frame=" << frame_w << "x"
              << frame_h << std::endl;

    DetectorOptions opt;
    try {
        opt.layer = ObjectLayerConfig::load(args.objects, EWORLD_SOURCE_DIR);
    } catch (const std::exception& e) {
        die(std::string("[aerial] ") + e.what());
    }
    if (!args.model.empty()) opt.layer.coco_model = args.model;
    if (!args.world_model.empty()) opt.layer.world_model = args.world_model;
    if (args.conf >= 0) opt.layer.coco_conf = args.conf;
    opt.imgsz = std::max(frame_w, frame_h);
    opt.ocr = args.ocr;
    opt.workers = args.threads;
    for (const auto& m : {opt.layer.coco_model, opt.layer.world_model})
        if (!m.empty() && !fs::is_regular_file(m)) die("model not found: " + m + " (run.sh exports it)");
    std::cout << "[aerial] objects: " << fs::path(opt.layer.coco_model).filename().string()
              << (opt.layer.world_model.empty() ? "" : " + " + fs::path(opt.layer.world_model).filename().string())
              << "; dedup -> one "
              << (args.ocr == "florence" ? "HaloCore Florence-2" : args.ocr == "tesseract" ? "Tesseract" : "no")
              << " read per object" << std::endl;
    std::shared_ptr<AerialDetector> detector;
    try {
        detector = std::make_shared<AerialDetector>(opt);
    } catch (const std::exception& e) {
        die(std::string("[aerial] ") + e.what());
    }

    ehc::media::StreamingVideoConfig cfg;
    cfg.dim = kDim;
    cfg.k = kK;
    cfg.frame_w = frame_w;
    cfg.frame_h = frame_h;
    cfg.seed = kSeed;
    cfg.color_mode = ehc::media::ColorMode::rgb;
    cfg.dedup_mode = ehc::media::DedupMode::phash_plus_vsa;
    cfg.stage2_pairwise = true;
    cfg.max_items_per_frame = 40;
    cfg.detector_token_seed = kTokenSeed;
    cfg.detectors = {detector};

    const json config_doc{
        {"dim", cfg.dim}, {"k", cfg.k}, {"frame_w", frame_w}, {"frame_h", frame_h}, {"seed", cfg.seed},
        {"target_fps", args.fps}, {"color_mode", "rgb"}, {"dedup_mode", "phash_plus_vsa"},
        {"stage2_pairwise", true}, {"max_items_per_frame", cfg.max_items_per_frame},
        {"lsh_num_tables", kNumTables}, {"lsh_hash_size", kHashSize},
        {"detector_cmd", "aerial_encode --objects " + args.objects + " --ocr " + args.ocr},
        {"detector_format", "objects+dedup+" + args.ocr},
        {"object_layer", opt.layer.to_json()},
        {"detector_token_seed", kTokenSeed},
    };
    std::ofstream(args.out_dir / "config.json") << config_doc.dump(2);

    ehc::index::BSCLSHIndex lsh(cfg.dim, cfg.k, kNumTables, kHashSize);
    std::ofstream vid_log(args.out_dir / "videos.jsonl"), frame_log(args.out_dir / "frames.jsonl"),
        overlay_log(args.out_dir / "overlay.jsonl"), object_log(args.out_dir / "objects.jsonl"),
        triple_log(args.out_dir / "triples.jsonl"), enc_log(args.out_dir / "encode_log.jsonl");

    std::vector<ehc::core::SparseVector> batch_vecs;
    std::vector<int64_t> batch_ids;
    int64_t next_row = 0;
    int n_ok = 0, n_fail = 0;
    long total_survivors = 0, total_input_frames = 0, total_objects = 0;
    const double t_start = now_s();

    for (size_t vi = 0; vi < sources.size(); ++vi) {
        const uint64_t vid_id = vi + 1;
        const fs::path& src = sources[vi];
        const double t0 = now_s();
        json vmeta;
        std::vector<ehc::media::EncodedFrame> encoded;
        ehc::media::StreamingVideoEncoder::Stats st;
        int n_input = 0;
        try {
            vmeta = video_meta(src.string());
            vmeta["video_id"] = vid_id;
            ehc::media::StreamingVideoEncoder enc(cfg);
            enc.begin_video(vid_id, vmeta["fps"].get<double>());
            ehc::media::FFmpegFramePipe pipe(src.string(), args.fps, frame_w, frame_h);
            while (auto f = pipe.next()) {
                enc.on_frame_rgb(f->data, f->h, f->w, f->pts_ms, f->frame_idx);
                ++n_input;
            }
            encoded = enc.finalize();
            st = enc.stats();
        } catch (const std::exception& e) {
            ++n_fail;
            enc_log << json{{"video_id", vid_id}, {"source", src.string()}, {"error", e.what()}}.dump() << "\n";
            std::cout << "[aerial] FAIL #" << vid_id << " " << src.filename().string() << ": " << e.what() << std::endl;
            continue;
        }
        const double wall_ms = (now_s() - t0) * 1000.0;

        std::vector<FrameRecord> records = detector->records();
        std::vector<ObjectEntry> objects = detector->objects();
        if (records.size() != encoded.size()) {
            // Stage-2 and detect() run on the same survivor list. A mismatch
            // means the stash was not taken. Keep the tags, skip the boxes.
            std::cerr << "[aerial] " << src.filename().string() << ": " << encoded.size() << " survivors, "
                      << records.size() << " overlay records\n";
            records.clear();
            objects.clear();
        }

        vid_log << vmeta.dump() << "\n";
        // The objects that become facts (config objects.min_frames / min_confidence).
        std::set<long> facts;
        for (const auto& o : objects)
            if (opt.layer.corroborated(o)) facts.insert(o.object_id);
        std::vector<int64_t> row_of(encoded.size());
        for (size_t i = 0; i < encoded.size(); ++i) {
            auto& ef = encoded[i];
            const int64_t row_id = row_of[i] = next_row++;
            frame_log << json{{"video_id", ef.meta.video_id}, {"frame_idx", ef.meta.frame_idx},
                              {"ts_ms", ef.meta.ts_ms}, {"row_id", row_id}, {"tags", ef.tags}}.dump()
                      << "\n";
            if (!records.empty()) {
                const auto& rec = records[i];
                json boxes = json::array();
                for (const auto& b : rec.boxes) boxes.push_back(box_json(b, facts.count(b.object_id) > 0));
                overlay_log << json{{"video_id", ef.meta.video_id}, {"source_path", vmeta["source_path"]},
                                    {"frame_idx", ef.meta.frame_idx}, {"ts_ms", ef.meta.ts_ms},
                                    {"row_id", row_id}, {"frame_w", frame_w}, {"frame_h", frame_h},
                                    {"src_w", vmeta["src_w"]}, {"src_h", vmeta["src_h"]},
                                    {"scene", rec.scene}, {"boxes", boxes}}.dump()
                            << "\n";
            }
        }
        // The object index: one row per deduplicated object, keyed by object_id.
        for (const auto& o : objects) {
            json seen = json::array();
            for (int s : o.survivors)
                seen.push_back({{"row_id", row_of[static_cast<size_t>(s)]},
                                {"frame_idx", encoded[static_cast<size_t>(s)].meta.frame_idx},
                                {"ts_ms", encoded[static_cast<size_t>(s)].meta.ts_ms}});
            const auto& best = encoded[static_cast<size_t>(o.best_survivor)].meta;
            object_log << json{{"object_id", o.object_id}, {"video_id", vid_id}, {"class", o.cls},
                               {"source", o.source}, {"confidence", o.confidence}, {"ocr", o.ocr},
                               {"fact", facts.count(o.object_id) > 0},
                               {"host_id", o.host_id >= 0 ? json(o.host_id) : json(nullptr)},
                               {"first_ts_ms", seen.front()["ts_ms"]}, {"last_ts_ms", seen.back()["ts_ms"]},
                               {"best", {{"frame_idx", best.frame_idx}, {"ts_ms", best.ts_ms},
                                         {"xyxy", {o.best_xyxy[0], o.best_xyxy[1], o.best_xyxy[2], o.best_xyxy[3]}}}},
                               {"frames", seen}}.dump()
                       << "\n";
        }
        write_triples(triple_log, src, vmeta, encoded, records, n_input, args.fps, opt.layer, facts);
        for (size_t i = 0; i < encoded.size(); ++i) {
            batch_vecs.push_back(std::move(encoded[i].vec));
            batch_ids.push_back(row_of[i]);
        }

        enc_log << json{{"video_id", vid_id}, {"source", src.filename().string()}, {"wall_ms", wall_ms},
                        {"input_frames", n_input}, {"frames_seen", st.frames_seen},
                        {"dropped_phash", st.dropped_phash}, {"dropped_vsa", st.dropped_vsa},
                        {"dropped_stage2", st.dropped_stage2}, {"survivors_kept", encoded.size()},
                        {"objects", objects.size()}, {"detect_seconds", detector->timing()},
                        {"detector_tokens_emitted", st.detector_tokens_emitted},
                        {"detector_tokens_dropped", st.detector_tokens_dropped}}.dump()
                << "\n";
        ++n_ok;
        total_survivors += static_cast<long>(encoded.size());
        total_input_frames += n_input;
        total_objects += static_cast<long>(objects.size());
        char line[512];
        std::snprintf(line, sizeof line,
                      "[aerial] #%llu/%zu %-40.40s in=%5d kept=%4zu objects=%4zu tags=%d dropped=%d wall=%8.0fms",
                      static_cast<unsigned long long>(vid_id), sources.size(), src.filename().string().c_str(),
                      n_input, encoded.size(), objects.size(), st.detector_tokens_emitted,
                      st.detector_tokens_dropped, wall_ms);
        std::cout << line << std::endl;
    }

    if (!batch_vecs.empty()) lsh.add_items(std::move(batch_vecs), batch_ids);
    lsh.save((args.out_dir / "frames.lsh").string());
    vid_log.close();
    frame_log.close();
    overlay_log.close();
    object_log.close();
    triple_log.close();
    enc_log.close();

    const json summary{
        {"videos_attempted", sources.size()}, {"videos_ok", n_ok}, {"videos_failed", n_fail},
        {"total_input_frames", total_input_frames}, {"total_survivors", total_survivors},
        {"total_objects", total_objects}, {"elapsed_sec", now_s() - t_start}, {"lsh_size", lsh.size()},
        {"coco_model", opt.layer.coco_model}, {"world_model", opt.layer.world_model}, {"ocr", args.ocr},
    };
    std::ofstream(args.out_dir / "encode_summary.json") << summary.dump(2);
    std::cout << "[aerial] DONE  survivors=" << total_survivors << "  objects=" << total_objects
              << "  lsh=" << lsh.size() << std::endl;
    return 0;
}
