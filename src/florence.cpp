// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#include "florence.hpp"

#include "doc/door_abi.h"   // HaloCore cpp/src: the one C call into the door

#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>

namespace eworld {
namespace fs = std::filesystem;

namespace {
constexpr const char* kDoorLib      = "/opt/mpl/lib/libhalocore_door.dylib";
constexpr const char* kDefaultModel = "/opt/mpl/models/florence-2-base-onnx";

std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return v && *v ? v : fallback;
}
}  // namespace

HaloFlorence::HaloFlorence(const std::string& model_dir) {
    const std::string lib = env_or("HALOCORE_DOOR_LIB", kDoorLib);
    if (!fs::is_regular_file(lib)) throw std::runtime_error("HaloCore door not found: " + lib);
    const std::string model = !model_dir.empty() ? model_dir : env_or("HALOCORE_FLORENCE_DIR", kDefaultModel);
    if (!fs::is_regular_file(fs::path(model) / "onnx" / "vision_encoder_int8.onnx"))
        throw std::runtime_error("HaloCore Florence-2 not found: " + model);
    ::setenv("HALOCORE_FLORENCE_DIR", model.c_str(), 1);
    ::setenv("HALOCORE_FIGURES_QUANT", "int8", 0);
    lib_ = ::dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib_) throw std::runtime_error(std::string("cannot open the HaloCore door: ") + ::dlerror());
    fn_ = ::dlsym(lib_, "halocore_door_run");
    if (!fn_) throw std::runtime_error("halocore_door_run missing from " + lib);
}

HaloFlorence::~HaloFlorence() {
    if (lib_) ::dlclose(lib_);
}

std::map<std::string, std::string> HaloFlorence::read_many(const std::vector<std::string>& image_paths) {
    std::map<std::string, std::string> by_stem;
    std::vector<const char*> paths;
    for (const auto& p : image_paths)
        if (!p.empty()) paths.push_back(p.c_str());
    if (paths.empty()) return by_stem;
    std::cout << "[aerial] halocore florence " << paths.size() << " frame(s)" << std::endl;

    std::error_code ec;
    fs::path parent = fs::temp_directory_path(ec) / ("eworld_florence_" + std::to_string(::getpid()));
    fs::remove_all(parent, ec);
    fs::create_directories(parent, ec);
    const fs::path store = parent / "store";

    HalocoreDoorRequest req{};
    req.abi      = HALOCORE_DOOR_ABI_VERSION;
    req.inputs   = paths.data();
    req.n_inputs = paths.size();
    const std::string store_s = store.string();
    req.jsonl    = store_s.c_str();
    req.figures  = 1;
    HalocoreDoorResult res{};
    const int rc = reinterpret_cast<halocore_door_run_fn>(fn_)(&req, &res);

    // rc 3 (written, a picture not read) is a failure too: a partial batch is not trusted.
    if (rc != 0 || !fs::is_regular_file(store)) {
        const std::string msg = res.error[0] ? res.error : std::to_string(rc);
        std::cout << "[aerial] halocore florence failed: " << msg << std::endl;
    } else {
        std::ifstream in(store);
        for (std::string line; std::getline(in, line);) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            nlohmann::json rec;
            try { rec = nlohmann::json::parse(line); } catch (const std::exception&) { continue; }
            const std::string stem = rec.value("doc", "");
            const std::string text = rec.value("text", "");
            if (stem.empty() || text.empty()) continue;
            auto& prev = by_stem[stem];
            prev = prev.empty() ? text : prev + "\n" + text;
        }
    }
    fs::remove_all(parent, ec);
    fs::remove_all(fs::path(store_s + ".figures"), ec);
    return by_stem;
}

}  // namespace eworld
