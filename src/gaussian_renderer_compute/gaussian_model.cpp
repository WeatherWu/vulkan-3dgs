#include "gaussian_model.hpp"
#include "utils/file_utils.hpp"
#include "utils/logger.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cmath>

namespace vk_gs {

// === SHColor实现 ===
namespace {

glm::vec3 restore3DGSColor(const glm::vec3& sh_color) {
    // Match the standard 3DGS SH color restore: add 0.5 and clamp only the lower bound.
    return glm::max(sh_color + glm::vec3(0.5f), glm::vec3(0.0f));
}

} // namespace

glm::vec3 SHColor::getColor(const glm::vec3& view_direction) const {
    // 球谐函数评估（3阶SH，共16个基函数）
    // 参考: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    
    const float SH_C0 = 0.28209479177387814f;
    const float SH_C1 = 0.4886025119029199f;
    const float SH_C2[] = {
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f
    };
    const float SH_C3[] = {
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f
    };
    
    glm::vec3 result = sh0 * SH_C0;
    
    if (view_direction.x == 0.0f && view_direction.y == 0.0f && view_direction.z == 0.0f) {
        return restore3DGSColor(result);
    }
    
    float x = view_direction.x;
    float y = view_direction.y;
    float z = view_direction.z;
    
    // 1阶SH (l=1, m=-1,0,1)
    result += SH_C1 * (-y * sh1[0] + z * sh1[1] - x * sh1[2]);
    
    // 2阶SH (l=2, m=-2,-1,0,1,2)
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    result += 
        SH_C2[0] * xy * sh2[0] +
        SH_C2[1] * yz * sh2[1] +
        SH_C2[2] * (2.0f * zz - xx - yy) * sh2[2] +
        SH_C2[3] * xz * sh2[3] +
        SH_C2[4] * (xx - yy) * sh2[4];
    
    // 3阶SH (l=3, m=-3,-2,-1,0,1,2,3)
    result +=
        SH_C3[0] * y * (3.0f * xx - yy) * sh3[0] +
        SH_C3[1] * xy * z * sh3[1] +
        SH_C3[2] * y * (4.0f * zz - xx - yy) * sh3[2] +
        SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh3[3] +
        SH_C3[4] * x * (4.0f * zz - xx - yy) * sh3[4] +
        SH_C3[5] * z * (xx - yy) * sh3[5] +
        SH_C3[6] * x * (xx - 3.0f * yy) * sh3[6];
    
    return restore3DGSColor(result);
}

// === 文件格式检测 ===
static std::string detectFileFormat(const std::string& filename) {
    // 根据文件扩展名检测格式
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return "unknown";
    }
    
    std::string ext = filename.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "ply") {
        return "ply";
    } else if (ext == "splat" || ext == "gs") {
        return "splat";
    } else if (ext == "json") {
        return "json";
    }
    
    return "unknown";
}

// === PLY文件加载 ===
bool GaussianModel::loadFromPLY(const std::string& filename) {
    std::string resolvedPath = FileUtils::resolvePath(filename);
    
    LOG_INFO("Loading PLY file: {}", resolvedPath);
    
    std::ifstream file(resolvedPath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open PLY file: {}", resolvedPath);
        return false;
    }
    
    // 解析PLY头部
    uint32_t vertex_count = 0;
    if (!parsePLYHeader(file, vertex_count)) {
        LOG_ERROR("Failed to parse PLY header: {}", resolvedPath);
        return false;
    }
    
    LOG_INFO("PLY file contains {} Gaussian points", vertex_count);
    
    // 清空现有数据
    clear();
    
    // 读取顶点数据
    points_.reserve(vertex_count);
    for (uint32_t i = 0; i < vertex_count; ++i) {
        GaussianPoint point;
        if (!parsePLYVertex(file, point)) {
            LOG_ERROR("Failed to read vertex {}", i);
            return false;
        }
        
        point.update_precomputed_data();
        points_.push_back(point);
        
        if (point.active) {
            active_indices_.push_back(points_.size() - 1);
        }
    }
    
    // 更新包围盒和统计信息
    updateBoundingBoxVolume();
    needs_update_ = false;
    
    LOG_INFO("Successfully loaded {} Gaussian points from PLY file", points_.size());
    return true;
}

bool GaussianModel::parsePLYHeader(std::ifstream& file, uint32_t& vertex_count) {
    // 清空之前的属性列表
    ply_properties_.clear();
    
    // 读取并验证PLY魔术数字
    std::string line;
    std::getline(file, line);
    
    if (line.find("ply") == std::string::npos && 
        line.find("PLY") == std::string::npos) {
        LOG_ERROR("Invalid PLY file: missing 'ply' magic number");
        return false;
    }
    
    bool is_binary = false;
    bool header_end = false;
    
    // 解析头部每一行
    while (std::getline(file, line)) {
        // 移除回车符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        // 检查头部结束标记
        if (line == "end_header") {
            header_end = true;
            break;
        }
        
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;
        
        if (keyword == "format") {
            std::string format_type;
            iss >> format_type;
            if (format_type == "binary_little_endian") {
                is_binary = true;
            } else if (format_type == "ascii") {
                is_binary = false;
            } else {
                LOG_WARN("Unknown PLY format: {}, assuming binary", format_type);
                is_binary = true;
            }
        } else if (keyword == "element" || keyword == "ELEMENT") {
            std::string element_name;
            uint32_t count;
            iss >> element_name >> count;
            
            if (element_name == "vertex" || element_name == "VERTEX") {
                vertex_count = count;
            }
        } else if (keyword == "property" || keyword == "PROPERTY") {
            // 解析属性定义
            std::string type, name;
            iss >> type >> name;
            
            if (!name.empty()) {
                PLYProperty prop;
                prop.name = name;
                prop.type = type;
                ply_properties_.push_back(prop);
                
                LOG_DEBUG("Found property: {} ({})", name, type);
            }
        }
    }
    
    if (!header_end) {
        LOG_ERROR("Invalid PLY file: missing 'end_header'");
        return false;
    }
    
    if (vertex_count == 0) {
        LOG_ERROR("Invalid PLY file: no vertex element found");
        return false;
    }
    
    LOG_INFO("Parsed {} properties from PLY header", ply_properties_.size());
    
    return true;
}

bool GaussianModel::parsePLYVertex(std::ifstream& file, GaussianPoint& point) const {
    // 3DGS PLY格式解析，包含完整的SH系数
    // 根据头部解析的属性列表动态读取和跳过数据
    
    if (ply_properties_.empty()) {
        LOG_ERROR("No properties parsed from PLY header");
        return false;
    }
    
    // 遍历所有属性，只读取我们需要的，跳过不需要的
    for (const auto& prop : ply_properties_) {
        size_t prop_size = prop.getSize();
        
        // 位置坐标 (x, y, z)
        if (prop.name == "x") {
            float x;
            file.read(reinterpret_cast<char*>(&x), prop_size);
            point.position.x = x;
        }
        else if (prop.name == "y") {
            float y;
            file.read(reinterpret_cast<char*>(&y), prop_size);
            point.position.y = y;
        }
        else if (prop.name == "z") {
            float z;
            file.read(reinterpret_cast<char*>(&z), prop_size);
            point.position.z = z;
        }
        // 0阶SH系数 (DC项)
        else if (prop.name == "f_dc_0") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.color.sh0.x = val;
        }
        else if (prop.name == "f_dc_1") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.color.sh0.y = val;
        }
        else if (prop.name == "f_dc_2") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.color.sh0.z = val;
        }
        // 1-3阶SH系数 (f_rest_0 到 f_rest_44)
        else if (prop.name.find("f_rest_") == 0) {
            // 提取索引号
            int idx = std::stoi(prop.name.substr(7));
            
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            
            // 3DGS PLY 的 f_rest 通常按颜色通道优先展开：
            // f_rest_0..14   -> R channel, SH bases 1..15
            // f_rest_15..29  -> G channel, SH bases 1..15
            // f_rest_30..44  -> B channel, SH bases 1..15
            int channel = idx / 15;
            int base_idx = idx % 15;
            
            if (base_idx < 3) {
                // 1阶SH (3个基函数)
                if (channel == 0) point.color.sh1[base_idx].x = val;
                else if (channel == 1) point.color.sh1[base_idx].y = val;
                else if (channel == 2) point.color.sh1[base_idx].z = val;
            }
            else if (base_idx < 8) {
                // 2阶SH (5个基函数)
                int sh2_idx = base_idx - 3;
                if (channel == 0) point.color.sh2[sh2_idx].x = val;
                else if (channel == 1) point.color.sh2[sh2_idx].y = val;
                else if (channel == 2) point.color.sh2[sh2_idx].z = val;
            }
            else if (base_idx < 15) {
                // 3阶SH (7个基函数)
                int sh3_idx = base_idx - 8;
                if (channel == 0) point.color.sh3[sh3_idx].x = val;
                else if (channel == 1) point.color.sh3[sh3_idx].y = val;
                else if (channel == 2) point.color.sh3[sh3_idx].z = val;
            }
        }
        // 不透明度
        else if (prop.name == "opacity") {
            float opacity;
            file.read(reinterpret_cast<char*>(&opacity), prop_size);
            point.alpha = 1.0f / (1.0f + std::exp(-opacity)); // sigmoid函数
        }
        // 缩放（对数空间）
        else if (prop.name == "scale_0") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.scale.x = std::exp(val);
        }
        else if (prop.name == "scale_1") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.scale.y = std::exp(val);
        }
        else if (prop.name == "scale_2") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.scale.z = std::exp(val);
        }
        // 旋转四元数
        else if (prop.name == "rot_0" || prop.name == "q w") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.rotation.w = val;
        }
        else if (prop.name == "rot_1" || prop.name == "q x") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.rotation.x = val;
        }
        else if (prop.name == "rot_2" || prop.name == "q y") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.rotation.y = val;
        }
        else if (prop.name == "rot_3" || prop.name == "q z") {
            float val;
            file.read(reinterpret_cast<char*>(&val), prop_size);
            point.rotation.z = val;
        }
        // 跳过不需要的属性（如法向量nx, ny, nz等）
        else {
            // 跳过该属性的数据
            file.seekg(prop_size, std::ios::cur);
            LOG_DEBUG("Skipping unused property: {}", prop.name);
        }
    }
    
    // 归一化四元数
    point.rotation = glm::normalize(point.rotation);
    
    // 构建协方差矩阵（从缩放和旋转）
    glm::mat3 S = glm::mat3(
        point.scale.x, 0.0f, 0.0f,
        0.0f, point.scale.y, 0.0f,
        0.0f, 0.0f, point.scale.z
    );
    
    glm::mat3 R = glm::mat3_cast(point.rotation);
    point.covariance = R * S * S * glm::transpose(R);
    
    return file.good();
}

// === 通用文件加载（自动检测格式）===
bool GaussianModel::loadFromFile(const std::string& filename) {
    std::string format = detectFileFormat(filename);
    
    LOG_INFO("Detected file format: {}", format);
    
    if (format == "ply") {
        return loadFromPLY(filename);
    } else if (format == "splat" || format == "gs") {
        // TODO: 实现splat/gs格式加载
        LOG_WARN("SPLAT/GS format loading not yet implemented");
        return false;
    } else if (format == "json") {
        // TODO: 实现JSON格式加载
        LOG_WARN("JSON format loading not yet implemented");
        return false;
    } else {
        LOG_ERROR("Unsupported file format: {}", format);
        return false;
    }
}

// === PLY文件导出 ===
bool GaussianModel::exportToPLY(const std::string& filename) const {
    LOG_INFO("Exporting to PLY file: {}", filename);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to create PLY file: {}", filename);
        return false;
    }
    
    // 写入PLY头部
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << points_.size() << "\n";
    
    // 定义属性（包含完整的SH系数）
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    
    // 1-3阶SH系数 (45个)
    for (int i = 0; i < 45; ++i) {
        file << "property float f_rest_" << i << "\n";
    }
    
    file << "property float opacity\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "end_header\n";
    
    // 写入顶点数据
    for (const auto& point : points_) {
        // 位置
        file.write(reinterpret_cast<const char*>(&point.position.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.position.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.position.z), sizeof(float));
        
        // 0阶SH系数 (DC项)
        float f_dc[3] = {
            point.color.sh0.x,
            point.color.sh0.y,
            point.color.sh0.z
        };
        file.write(reinterpret_cast<const char*>(f_dc), 3 * sizeof(float));
        
        // 1-3阶SH系数 (45个系数)，按3DGS通道优先顺序导出。
        float f_rest[45];
        int idx = 0;
        for (int channel = 0; channel < 3; ++channel) {
            for (int i = 0; i < 3; ++i) {
                f_rest[idx++] = point.color.sh1[i][channel];
            }
            for (int i = 0; i < 5; ++i) {
                f_rest[idx++] = point.color.sh2[i][channel];
            }
            for (int i = 0; i < 7; ++i) {
                f_rest[idx++] = point.color.sh3[i][channel];
            }
        }
        file.write(reinterpret_cast<const char*>(f_rest), 45 * sizeof(float));
        
        // 不透明度（逆sigmoid）
        float opacity = std::log(point.alpha / (1.0f - point.alpha + 1e-8f) + 1e-8f);
        file.write(reinterpret_cast<const char*>(&opacity), sizeof(float));
        
        // 缩放（对数空间）
        float scale[3] = {
            std::log(point.scale.x + 1e-8f),
            std::log(point.scale.y + 1e-8f),
            std::log(point.scale.z + 1e-8f)
        };
        file.write(reinterpret_cast<const char*>(scale), 3 * sizeof(float));
        
        // 旋转四元数
        file.write(reinterpret_cast<const char*>(&point.rotation.w), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.rotation.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.rotation.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&point.rotation.z), sizeof(float));
    }
    
    LOG_INFO("Successfully exported {} points to PLY file", points_.size());
    return true;
}

// === 保存文件（自动选择格式）===
bool GaussianModel::exportToFile(const std::string& filename) const {
    std::string format = detectFileFormat(filename);
    
    if (format == "ply") {
        return exportToPLY(filename);
    } else {
        LOG_ERROR("Unsupported export format: {}", format);
        return false;
    }
}

// === 其他方法实现 ===

void GaussianModel::updateBoundingBoxVolume() {
    if (points_.empty()) {
        bounding_box_min_ = glm::vec3(0.0f);
        bounding_box_max_ = glm::vec3(0.0f);
        center_ = glm::vec3(0.0f);
        radius_ = 0.0f;
        return;
    }
    
    // 初始化包围盒
    bounding_box_min_ = points_[0].position;
    bounding_box_max_ = points_[0].position;
    
    // 遍历所有点找到最小和最大值
    for (const auto& point : points_) {
        bounding_box_min_ = glm::min(bounding_box_min_, point.position);
        bounding_box_max_ = glm::max(bounding_box_max_, point.position);
    }
    
    // 计算中心和半径
    center_ = (bounding_box_min_ + bounding_box_max_) * 0.5f;
    radius_ = glm::length(bounding_box_max_ - bounding_box_min_) * 0.5f;
}

void GaussianModel::buildSpatialIndex() {
    // TODO: 实现空间索引结构（如八叉树、BVH等）
    updateBoundingBoxVolume();
    LOG_INFO("Spatial index built (simplified implementation)");
}

void GaussianModel::translate(const glm::vec3& translation) {
    for (auto& point : points_) {
        point.position += translation;
    }
    updateBoundingBoxVolume();
    needs_update_ = true;
}

void GaussianModel::rotate(const glm::quat& rotation) {
    for (auto& point : points_) {
        glm::vec3 relative_pos = point.position - center_;
        point.position = center_ + rotation * relative_pos;
        point.rotation = rotation * point.rotation;
    }
    updateBoundingBoxVolume();
    needs_update_ = true;
}

void GaussianModel::scale(const glm::vec3& scaling) {
    for (auto& point : points_) {
        glm::vec3 relative_pos = point.position - center_;
        point.position = center_ + relative_pos * scaling;
        point.scale *= scaling;
    }
    updateBoundingBoxVolume();
    needs_update_ = true;
}

void GaussianModel::setTransform(const glm::mat4& transform) {
    // TODO: 实现完整的变换矩阵应用
    LOG_WARN("set_transform not fully implemented");
}

// void GaussianModel::pruneLowAlphaPoints(float threshold) {
//     size_t before_count = points_.size();
    
//     auto it = std::remove_if(points_.begin(), points_.end(),
//         [threshold](const GaussianPoint& point) {
//             return point.alpha < threshold;
//         });
    
//     points_.erase(it, points_.end());
//     rebuildActiveIndices();
    
//     size_t removed = before_count - points_.size();
//     if (removed > 0) {
//         LOG_INFO("Pruned {} low-alpha points", removed);
//     }
// }

std::vector<float> GaussianModel::get_vertex_data() const {
    std::vector<float> data;
    data.reserve(points_.size() * 14); // 每个点14个float
    
    for (const auto& point : points_) {
        // 位置
        data.push_back(point.position.x);
        data.push_back(point.position.y);
        data.push_back(point.position.z);
        
        // 颜色（从SH的sh0获取）
        data.push_back(point.color.sh0.x);
        data.push_back(point.color.sh0.y);
        data.push_back(point.color.sh0.z);
        
        // 缩放
        data.push_back(point.scale.x);
        data.push_back(point.scale.y);
        data.push_back(point.scale.z);
        
        // 旋转
        data.push_back(point.rotation.x);
        data.push_back(point.rotation.y);
        data.push_back(point.rotation.z);
        data.push_back(point.rotation.w);
        
        // Alpha
        data.push_back(point.alpha);
    }
    
    return data;
}

std::vector<uint32_t> GaussianModel::get_index_data() const {
    // 对于点云渲染，通常不需要索引
    std::vector<uint32_t> indices(points_.size());
    for (size_t i = 0; i < points_.size(); ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }
    return indices;
}

std::vector<float> GaussianModel::get_active_vertex_data() const {
    std::vector<float> data;
    data.reserve(active_indices_.size() * 14);
    
    for (size_t idx : active_indices_) {
        const auto& point = points_[idx];
        
        data.push_back(point.position.x);
        data.push_back(point.position.y);
        data.push_back(point.position.z);
        
        data.push_back(point.color.sh0.x);
        data.push_back(point.color.sh0.y);
        data.push_back(point.color.sh0.z);
        
        data.push_back(point.scale.x);
        data.push_back(point.scale.y);
        data.push_back(point.scale.z);
        
        data.push_back(point.rotation.x);
        data.push_back(point.rotation.y);
        data.push_back(point.rotation.z);
        data.push_back(point.rotation.w);
        
        data.push_back(point.alpha);
    }
    
    return data;
}

std::vector<uint32_t> GaussianModel::get_active_index_data() const {
    std::vector<uint32_t> indices(active_indices_.size());
    for (size_t i = 0; i < active_indices_.size(); ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }
    return indices;
}

} // namespace vk_gs
