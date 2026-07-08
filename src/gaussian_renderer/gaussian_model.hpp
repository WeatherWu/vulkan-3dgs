#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <memory>
#include <string>
#include <fstream>

namespace vulkan3DGS {

// PLY属性描述
struct PLYProperty {
    std::string name;
    std::string type;  // "float", "int", "uchar" 等
    
    size_t getSize() const {
        if (type == "float" || type == "double") return sizeof(float);
        if (type == "int" || type == "uint") return sizeof(int32_t);
        if (type == "uchar" || type == "char") return sizeof(uint8_t);
        if (type == "short" || type == "ushort") return sizeof(int16_t);
        return sizeof(float); // 默认
    }
};

struct SHColor {
    glm::vec3 sh0;        // 0阶SH（基础颜色）
    glm::vec3 sh1[3];           // 1阶SH系数（3个方向）
    glm::vec3 sh2[5];           // 2阶SH系数（5个方向）  
    glm::vec3 sh3[7];           // 3阶SH系数（7个方向）
    
    glm::vec3 getColor(const glm::vec3& view_direction) const;
};

// 高斯点数据结构
struct GaussianPoint {
    glm::vec3 position;           // 位置 (x, y, z)
    SHColor color;                // SH颜色
    glm::vec3 scale;              // 缩放 (sx, sy, sz)
    glm::quat rotation;           // 旋转四元数
    float alpha;                  // 透明度/不透明度
    glm::mat3 covariance;         // 3x3协方差矩阵
    
    // 预计算渲染数据
    glm::mat3 inverse_covariance; // 逆协方差矩阵（用于渲染）
    float determinant;            // 行列式（用于渲染）
    
    // 优化参数
    float learning_rate = 0.01f;  // 学习率
    bool active = true;           // 是否激活
    
    GaussianPoint() 
        : position(0.0f), color(), scale(1.0f), 
          rotation(1.0f, 0.0f, 0.0f, 0.0f), alpha(1.0f),
          covariance(1.0f), inverse_covariance(1.0f), determinant(1.0f) {}
    
    // 更新预计算数据
    void update_precomputed_data() {
        // 计算逆协方差矩阵和行列式
        inverse_covariance = glm::inverse(covariance);
        determinant = glm::determinant(covariance);
    }
};

class GaussianModel {
private:
    std::vector<GaussianPoint> points_;          // 高斯点集合
    std::vector<size_t> active_indices_;         // 激活点的索引
    
    // 模型属性
    glm::vec3 bounding_box_min_;                 // 包围盒最小值
    glm::vec3 bounding_box_max_;                 // 包围盒最大值
    glm::vec3 center_;                           // 模型中心
    float radius_;                               // 包围球半径
    
    // 优化状态
    bool needs_update_ = true;                   // 是否需要更新
    uint32_t iteration_count_ = 0;               // 优化迭代次数
    
public:
    GaussianModel() 
        : bounding_box_min_(0.0f), bounding_box_max_(0.0f), 
          center_(0.0f), radius_(0.0f) {}
    
    ~GaussianModel() = default;
    
    void clear() {
        points_.clear();
        active_indices_.clear();
        needs_update_ = true;
    }
    
    // === 文件操作 ===
    bool loadFromFile(const std::string& filename);
    bool exportToFile(const std::string& filename) const;
    bool loadFromPLY(const std::string& filename);
    bool exportToPLY(const std::string& filename) const;
    
    // === 模型变换 ===
    void translate(const glm::vec3& translation);
    void rotate(const glm::quat& rotation);
    void scale(const glm::vec3& scaling);
    void setTransform(const glm::mat4& transform);
    
    // === 统计分析 ===
    glm::vec3 get_center() const { return center_; }
    float get_radius() const { return radius_; }
    glm::vec3 get_bounding_box_min() const { return bounding_box_min_; }
    glm::vec3 get_bounding_box_max() const { return bounding_box_max_; }
    
    // 统计信息
    struct Statistics {
        size_t total_points;
        size_t active_points;
        float average_alpha;
        float average_scale;
    };
    
    Statistics get_statistics() const {
        Statistics stats{};
        stats.total_points = points_.size();
        stats.active_points = active_indices_.size();
        
        if (!points_.empty()) {
            float alpha_sum = 0.0f;
            float scale_sum = 0.0f;
            glm::vec3 color_min(1.0f), color_max(0.0f);
            
            for (const auto& point : points_) {
                alpha_sum += point.alpha;
                scale_sum += glm::length(point.scale);
            }
            
            stats.average_alpha = alpha_sum / points_.size();
            stats.average_scale = scale_sum / points_.size();
        }
        
        return stats;
    }
    
    // === 渲染相关 ===
    // 获取用于渲染的顶点数据
    std::vector<float> get_vertex_data() const;
    
    // 获取用于渲染的索引数据
    std::vector<uint32_t> get_index_data() const;
    
    // 获取激活点的渲染数据（优化性能）
    std::vector<float> get_active_vertex_data() const;
    std::vector<uint32_t> get_active_index_data() const;
    
    // === 实用方法 ===
    void updateBoundingBoxVolume();
    void buildSpatialIndex();
    void normalizeColors();
    void clampParameters();
    
    // 状态检查
    bool isEmpty() const { return points_.empty(); }
    bool needsUpdate() const { return needs_update_; }
    void markUpdated() { needs_update_ = false; }
    
    // 迭代器支持
    auto begin() { return points_.begin(); }
    auto end() { return points_.end(); }
    auto begin() const { return points_.begin(); }
    auto end() const { return points_.end(); }
    
private:
    // === 基本操作 ===
    void addPoint(const GaussianPoint& point) {
        points_.push_back(point);
        points_.back().update_precomputed_data();
        if (point.active) {
            active_indices_.push_back(points_.size() - 1);
        }
        needs_update_ = true;
    }
    
    void removePoint(size_t index) {
        if (index < points_.size()) {
            points_.erase(points_.begin() + index);
            rebuildActiveIndices();
            needs_update_ = true;
        }
    }
    
    // 内部辅助方法
    void updatePointPrecomputedData(size_t index) {
        if (index < points_.size()) {
            points_[index].update_precomputed_data();
        }
    }
    
    void rebuildActiveIndices() {
        active_indices_.clear();
        for (size_t i = 0; i < points_.size(); ++i) {
            if (points_[i].active) {
                active_indices_.push_back(i);
            }
        }
    }
    
    void updateSpatialIndex() {
        // 简化的空间索引更新
        updateBoundingBoxVolume();
    }
    
    // 文件格式解析
    bool parsePLYHeader(std::ifstream& file, uint32_t& vertex_count);
    bool parsePLYVertex(std::ifstream& file, GaussianPoint& point) const;
    
    // PLY属性列表（在parsePLYHeader中填充）
    std::vector<PLYProperty> ply_properties_;
};

} // namespace vulkan3DGS