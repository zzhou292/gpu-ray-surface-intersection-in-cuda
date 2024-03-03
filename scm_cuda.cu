#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <vector>
#include <sstream>
#include <stdint.h>
#include <filesystem>
#include <iomanip> // For std::fixed and std::setprecision
#include "ray_tracing.cuh"
using namespace std;


struct ObjectState {
    float displacementX = 0.0f; // Total displacement along the X-axis
    float displacementZ = 0.0f; // Total displacement along the Z-axis
    float rotationAngleY = 0.0f; // Total rotation around the Y-axis
};

// =================== JSON added helper function =======================
void readMeshData(string fname, vector<float>& v, vector<int>& surf) {
    std::ifstream file(fname);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x,y,z;
            iss >> x >> y >> z;
            v.push_back(x);
            v.push_back(y);
            v.push_back(z);
        } else if (prefix == "f") {
            std::string vertex;
            for (int i = 0; i < 3; ++i) {
                iss >> vertex;
                size_t slash_pos = vertex.find('/');
                if (slash_pos != std::string::npos) {
                    vertex = vertex.substr(0, slash_pos);
                }
                int idx = std::stoi(vertex) - 1;
                surf.push_back(idx);
            }
        }
    }
}
void applyTransformation(std::vector<float>& vertices, ObjectState& state, float deltaDisplacementX, float deltaDisplacementZ) {
    // Update the state with the new transformations
    state.displacementX += deltaDisplacementX;
    state.displacementZ += deltaDisplacementZ;

    // Apply the updated transformations to each vertex
    for (size_t i = 0; i < vertices.size(); i += 3) {
        vertices[i] = vertices[i] + state.displacementX;
        vertices[i+1] = vertices[i + 1]; // Y-coordinate remains unchanged
        vertices[i+2] = vertices[i + 2] + state.displacementZ; // Apply continuous displacement
    }
}


void writeOBJFile(const std::string& filename, const std::vector<float>& vertices, const std::vector<int>& faces) {
    std::ofstream outFile(filename);

    if (!outFile) {
        std::cerr << "Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write vertices
    for (size_t i = 0; i < vertices.size(); i += 3) {
        outFile << "v " << vertices[i] << " " << vertices[i + 1] << " " << vertices[i + 2] << std::endl;
    }

    // Write faces
    for (size_t i = 0; i < faces.size(); i += 3) {
        outFile << "f " << faces[i] + 1 << " " << faces[i + 1] + 1 << " " << faces[i + 2] + 1 << std::endl;
    }

    outFile.close();
}

void writeTerrainCSVFile(const std::string& filename, const std::vector<float>& vertices) {
    std::ofstream outFile(filename);

    if (!outFile) {
        std::cerr << "Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(6);
    outFile << "X,Y,Z\n"; // Optional: Write column headers

    // Write vertices
    for (size_t i = 0; i < vertices.size(); i += 3) {
        outFile << vertices[i] << "," << vertices[i + 1] << "," << vertices[i + 2] << "\n";
    }

    outFile.close();
    std::cout << "Written vertices to " << filename << std::endl;
}

void writeVTKFile(const std::string& filename, const std::vector<float>& vertices, const std::vector<int>& faces) {
    std::ofstream outFile(filename);

    if (!outFile) {
        std::cerr << "Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // VTK header
    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "Mesh data\n"; // Title
    outFile << "ASCII\n";
    outFile << "DATASET UNSTRUCTURED_GRID\n";

    // Writing vertices (points)
    outFile << "POINTS " << vertices.size() / 3 << " float\n";
    for (size_t i = 0; i < vertices.size(); i += 3) {
        outFile << vertices[i] << " " << vertices[i + 1] << " " << vertices[i + 2] << "\n";
    }

    // Writing faces (cells)
    size_t numFaces = faces.size() / 3;
    outFile << "CELLS " << numFaces << " " << numFaces * 4 << "\n";
    for (size_t i = 0; i < faces.size(); i += 3) {
        // VTK expects cells to be defined by the number of points followed by the point indices
        outFile << "3 " << faces[i] << " " << faces[i + 1] << " " << faces[i + 2] << "\n";
    }

    // Writing cell types (5 corresponds to triangle in VTK)
    outFile << "CELL_TYPES " << numFaces << "\n";
    for (size_t i = 0; i < numFaces; i++) {
        outFile << "5\n"; // VTK_TRIANGLE
    }

    outFile.close();
}


int main() {

    std::filesystem::create_directories("output");

    // Define the range and resolution
    float start = -5.0f, end = 5.0f, resolution = 0.005f;
    int size = static_cast<int>((end - start) / resolution);

    vector<float> h_rayFrom;
    vector<float> h_rayTo;

    // Allocate and initialize vectors x, y, z on host
    std::vector<float> h_x(size), h_y(size), h_z(size);
    for (int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++){
            float x = start + i * resolution;
            float y = start + j * resolution;
            h_rayFrom.push_back(x);
            h_rayFrom.push_back(y);
            h_rayFrom.push_back(-0.12);
            h_rayTo.push_back(x);
            h_rayTo.push_back(y);
            h_rayTo.push_back(5.0);
        }
    }

    // read input data into host memory
    vector<float> h_vertices;
    vector<int> h_triangles;
    vector<int> valid_outcome_idx;
    vector<float> valid_outcome;
    readMeshData("cobra_wheel.obj", h_vertices, h_triangles);

    ObjectState objectState;
    for(int j = 0; j < 200; j++){

        vector<float> h_rayFrom_tol = h_rayFrom;
        for(int i = 0; i < h_rayFrom.size()/3; i++){
            h_rayFrom_tol[3*i+2] = h_rayFrom[3*i+2]-0.5;
        }


        // perform ray tracing
        gpu_trace_ray(h_vertices, h_triangles, h_rayFrom_tol, h_rayTo, valid_outcome_idx, valid_outcome);

        std::cout << "Number of intersections: " << valid_outcome_idx.size() << std::endl;

        // deform terrain
        for(int i = 0; i < valid_outcome_idx.size(); i++) {
            int temp_x = static_cast<int>((valid_outcome[3*i]-start) / resolution);
            int temp_y = static_cast<int>((valid_outcome[3*i+1]-start) / resolution);
            //std::cout << temp_x << " " << temp_y << std::endl;
            if(h_rayFrom[3*(temp_x*size+temp_y)+2] > valid_outcome[3*i+2]){
               h_rayFrom[3*(temp_x*size+temp_y)+2] = valid_outcome[3*i+2];
            }
            
        }

        // apply tire motion
        if(j<=10){
            applyTransformation(h_vertices, objectState, 0.001, -0.0002);
        }else{
            applyTransformation(h_vertices, objectState, 0.001, 0.0);
        }
        

        // Write the transformed object to a new OBJ file
        writeVTKFile("output/transformed_cobra_wheel_"+to_string(j)+".vtk", h_vertices, h_triangles);
        writeTerrainCSVFile("output/terrain_vertices_"+to_string(j)+".csv", h_rayFrom);
    }



    return 0;
}
