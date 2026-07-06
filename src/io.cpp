#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <io.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <geometry.hpp>

std::vector<Triangle> loadObj(const std::string& path) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;  // quads/n-gons become triangles automatically

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, config))
        throw std::runtime_error(reader.Error());

    const auto& attrib = reader.GetAttrib();
    std::vector<Triangle> tris;

    for (const auto& shape : reader.GetShapes()) {
        const auto& idx = shape.mesh.indices;
        for (size_t i = 0; i + 2 < idx.size(); i += 3) {
            auto vert = [&](size_t k) {
                int vi = idx[i + k].vertex_index;
                return Vec3{ attrib.vertices[3*vi + 0],
                             attrib.vertices[3*vi + 1],
                             attrib.vertices[3*vi + 2] };
            };
            Vec3 v0 = vert(0);
            Vec3 v1 = vert(1);
            Vec3 v2 = vert(2);

            // Calculate the geometric normal vector
            Vec3 norm = normalize(cross(v1 - v0, v2 - v0));
            tris.push_back({ v0, v1, v2, norm });
        }
    }
    return tris;
}
