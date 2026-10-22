#pragma once
#include <stdexcept>
#include <string_view>

namespace optimizer_candidate {
enum class Method { None, StreamAdam, SHColorReuse };
struct Description {
    std::string_view key, name, shader, layout, reportExtension;
};
constexpr Description describe(Method method) {
    switch (method) {
        case Method::StreamAdam:
            return {"r1", "stream_adam_v1",
                    "train_backward_2dgs_to_3dgs_optimizer_stream_adam.comp.spv",
                    "unchanged typed AoS, scoped float4 moments", ".s4_r1.json"};
        case Method::SHColorReuse:
            return {"sh_color_reuse", "sh_color_reuse_v1",
                    "train_backward_2dgs_to_3dgs_optimizer_sh_color_reuse.comp.spv",
                    "unchanged typed AoS, shared active color gradient and view direction",
                    ".sh_color_reuse.json"};
        default:
            return {"", "", "", "", ".s4_replay.json"};
    }
}
inline Method select(Method current, std::string_view option) {
    if (current != Method::None)
        throw std::runtime_error("Choose exactly one optimizer candidate; duplicates are rejected");
    if (option == "--candidate-r1") return Method::StreamAdam;
    if (option == "--candidate-sh-color-reuse") return Method::SHColorReuse;
    throw std::runtime_error("Unknown optimizer candidate");
}
} // namespace optimizer_candidate
