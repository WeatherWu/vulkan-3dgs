#pragma once
#include "training_pixel_culling_gpu_probe.hpp"

#include <map>
#include <set>

// Declared logical Workgroup storage only; NOT native allocation or occupancy.
inline std::vector<uint32_t> readRowShaderWords(const std::string& shader) {
    const auto path = std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders/" + shader + ".comp.spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamsize>(file.tellg());
    if (!file || bytes < 20 || bytes % 4) throw std::runtime_error("Invalid RowEmpty SPIR-V");
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), bytes);
    if (!file || words[0] != 0x07230203u) throw std::runtime_error("RowEmpty SPIR-V read failed");
    return words;
}
inline uint64_t inspectRowShaderWords(const std::vector<uint32_t>& words, bool candidate) {
    if (words.size() < 5u || words[0] != 0x07230203u)
        throw std::runtime_error("Invalid RowEmpty SPIR-V header");
    struct Type {
        uint32_t op;
        std::vector<uint32_t> args;
    };
    std::map<uint32_t, Type> types;
    std::map<uint32_t, uint32_t> constants, strides;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> offsets;
    std::vector<uint32_t> workgroup;
    std::set<uint32_t> laneBuiltins, relaxed;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> loads, variables;
    bool precise = false, workgroupSize = false, exp = false, log = false;
    for (size_t i = 5; i < words.size();) {
        const auto n = words[i] >> 16u, op = words[i] & 65535u;
        if (!n || n > words.size() - i) throw std::runtime_error("Invalid RowEmpty instruction");
        if (op >= 21u && op <= 32u && n >= 3u)
            types[words[i + 1]] = {op, {words.begin() + i + 2, words.begin() + i + n}};
        if (op == 43u && n == 4u) constants[words[i + 2]] = words[i + 3];
        if (op == 71u && n >= 3u) {
            precise |= words[i + 2] == 42u;
            if (words[i + 2] == 0u) relaxed.insert(words[i + 1]);
            if (n == 4u && words[i + 2] == 11u && words[i + 3] == 41u)
                laneBuiltins.insert(words[i + 1]);
            if (n == 4u && words[i + 2] == 6u) strides[words[i + 1]] = words[i + 3];
        }
        if (candidate && op == 72u && n >= 4u && words[i + 3] == 0u)
            throw std::runtime_error("RowEmpty forbids RelaxedPrecision struct members");
        if (candidate && op == 22u && n >= 3u && words[i + 2] != 32u)
            throw std::runtime_error("RowEmpty requires FP32");
        if (op == 61u && n >= 4u) loads[words[i + 2]] = {words[i + 1], words[i + 3]};
        if (op == 59u && n >= 4u) variables[words[i + 2]] = {words[i + 1], words[i + 3]};
        if (op == 12u && n >= 5u) {
            exp |= words[i + 4] == 27u;
            log |= words[i + 4] == 28u;
        }
        if (op == 72u && n == 5u && words[i + 3] == 35u)
            offsets[{words[i + 1], words[i + 2]}] = words[i + 4];
        if (op == 59u && n >= 4u && words[i + 3] == 4u) workgroup.push_back(words[i + 1]);
        if (op == 16u && n == 6u && words[i + 2] == 17u)
            workgroupSize = words[i + 3] == 16u && words[i + 4] == 16u && words[i + 5] == 1u;
        i += n;
    }
    if (!workgroupSize || (candidate && !precise))
        throw std::runtime_error("RowEmpty lost workgroup size/NoContraction; re-audit shader");
    if (candidate && (!exp || !log)) throw std::runtime_error("RowEmpty exp/log lowering changed");
    // glslang decorates ONLY the uint lane builtin and its direct loads as
    // RelaxedPrecision. Their values are [0,31] under the actual native32 gate.
    // No arithmetic result, float, Gaussian ID or other integer is exempted.
    const auto isUint32 = [&](uint32_t id) {
        return types.contains(id) && types.at(id).op == 21u &&
               types.at(id).args == std::vector<uint32_t>{32u, 0u};
    };
    const auto isLanePointer = [&](uint32_t id) {
        if (!laneBuiltins.contains(id) || !variables.contains(id) || variables.at(id).second != 1u)
            return false;
        const auto pointer = variables.at(id).first;
        return types.contains(pointer) && types.at(pointer).op == 32u &&
               types.at(pointer).args.size() == 2u && types.at(pointer).args[0] == 1u &&
               isUint32(types.at(pointer).args[1]);
    };
    if (candidate)
        for (auto id : relaxed)
            if (!isLanePointer(id) && !(loads.contains(id) && isUint32(loads.at(id).first) &&
                                        isLanePointer(loads.at(id).second)))
                throw std::runtime_error(
                    "RowEmpty forbids RelaxedPrecision except bounded uint lane builtin/loads");
    const auto size = [&](auto&& self, uint32_t id, uint32_t depth) -> uint64_t {
        if (depth > 32u || !types.contains(id)) throw std::runtime_error("Unknown Workgroup type");
        const auto& t = types.at(id);
        const auto& a = t.args;
        if ((t.op == 21u || t.op == 22u) && !a.empty() && a[0] == 32u) return 4;
        if ((t.op == 23u || t.op == 24u) && a.size() == 2u)
            return self(self, a[0], depth + 1) * a[1];
        if (t.op == 28u && a.size() == 2u && constants.contains(a[1])) {
            const auto element = self(self, a[0], depth + 1);
            return (strides.contains(id) ? strides.at(id) : element) * uint64_t(constants.at(a[1]));
        }
        if (t.op == 30u) {
            uint64_t end = 0;
            for (uint32_t m = 0; m < a.size(); ++m) {
                const auto start = offsets.contains({id, m}) ? offsets.at({id, m}) : end;
                end = std::max(end, start + self(self, a[m], depth + 1));
            }
            return end;
        }
        if (t.op == 32u && a.size() == 2u && a[0] == 4u) return self(self, a[1], depth + 1);
        throw std::runtime_error("Unsupported Workgroup type: re-audit declared resource budget");
    };
    uint64_t total = 0;
    for (auto id : workgroup)
        total += size(size, id, 0);
    if (!total || total > 1024u * 1024u) throw std::runtime_error("Invalid shared resource total");
    return total;
}
inline uint64_t rowShaderSharedBytes(const std::string& shader, bool candidate) {
    return inspectRowShaderWords(readRowShaderWords(shader), candidate);
}
inline void checkRowShaderAudit() {
    const auto words = readRowShaderWords("train_backward_vksplat_tensor_row_empty");
    if (inspectRowShaderWords(words, true) != 33668u)
        throw std::runtime_error("RowEmpty declared shared budget changed");
    const auto mustReject = [](const auto& changed) {
        try {
            (void)inspectRowShaderWords(changed, true);
        } catch (const std::runtime_error&) {
            return;
        }
        throw std::runtime_error("RowEmpty shader audit accepted a broken precision contract");
    };
    for (uint32_t opcode :
         {129u, 128u}) { // Float addition and integer arithmetic: neither may be relaxed.
        auto changed = words;
        bool found = false;
        for (size_t i = 5; i < words.size(); i += words[i] >> 16u)
            if ((words[i] & 65535u) == opcode) {
                changed.insert(changed.end(), {(3u << 16u) | 71u, words[i + 2], 0u});
                found = true;
                break;
            }
        if (!found) throw std::runtime_error("RowEmpty audit fixture lost arithmetic coverage");
        mustReject(changed);
    }
    auto noPrecise = words;
    for (size_t i = 5; i < noPrecise.size(); i += noPrecise[i] >> 16u)
        if ((noPrecise[i] & 65535u) == 71u && noPrecise[i] >> 16u >= 3u && noPrecise[i + 2] == 42u)
            noPrecise[i + 2] =
                0u; // A forbidden RelaxedPrecision also cannot replace NoContraction.
    mustReject(noPrecise);
}
