// xp_from_phi.cpp
// Translate Python Xp_mol to C++17 + MPI + Eigen, using CSV phix/phiy/phiz in ./_phi_/
// For each frame dump.*.txt, outputs Xp/Yp/Zp matrices to Xp5/x_..., y_..., z_...

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace fs = std::filesystem;

// ---------------- Configurable parameters ----------------
static const std::string BASE_DIR     = "./";
static const std::string INPUT_DIR    = "./_coord_/";
static const std::string PHI_DIR      = "./_phi_/";
static const std::string OUTPUT_DIR   = "./Xp5/";
static const int HEADER_LINES         = 9;     // lines to skip in dump files
static const int num_mol              = 100;   // molecules
static const int mol_beads            = 720;   // beads per molecule
// ---------------------------------------------------------

// Basic helpers
static inline bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
}
static inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && std::equal(suf.rbegin(), suf.rend(), s.rbegin());
}

// Discover frame files: INPUT_DIR/dump.*.txt (sorted)
std::vector<std::string> discover_frames_mpi(int rank) {
    std::vector<std::string> files;
    if (rank == 0) {
        if (!fs::exists(INPUT_DIR)) throw std::runtime_error("No input dir: " + INPUT_DIR);
        for (const auto& e : fs::directory_iterator(INPUT_DIR)) {
            if (!e.is_regular_file()) continue;
            const auto name = e.path().filename().string();
            if (starts_with(name, "dump.") && ends_with(name, ".txt"))
                files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    }
    // Broadcast filenames
    int nf = static_cast<int>(files.size());
    MPI_Bcast(&nf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) files.resize(nf);
    for (int i = 0; i < nf; ++i) {
        int len = (rank == 0) ? static_cast<int>(files[i].size()) : 0;
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) files[i].resize(len);
        MPI_Bcast(files[i].data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    return files;
}

// Read CSV into Eigen::MatrixXd (no header). Returns rows x cols matrix.
// We expect matrices to be 720x720.
Eigen::MatrixXd read_csv_matrix(const std::string& fname) {
    std::ifstream fin(fname);
    if (!fin) throw std::runtime_error("Cannot open CSV: " + fname);

    std::string line;
    std::vector<double> data;
    int cols = -1;
    int rows = 0;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string cell;
        int this_cols = 0;
        while (std::getline(ss, cell, ',')) {
            // trim spaces
            cell.erase(cell.begin(), std::find_if(cell.begin(), cell.end(), [](unsigned char ch){ return !std::isspace(ch); }));
            cell.erase(std::find_if(cell.rbegin(), cell.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), cell.end());
            if (!cell.empty())
                data.push_back(std::stod(cell));
            else
                data.push_back(0.0);
            ++this_cols;
        }
        if (cols < 0) cols = this_cols;
        else if (cols != this_cols) throw std::runtime_error("Inconsistent columns in " + fname);
        ++rows;
    }
    if (cols <= 0 || rows <= 0) throw std::runtime_error("Empty CSV: " + fname);
    Eigen::MatrixXd M(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            M(r, c) = data[r * cols + c];
    return M;
}

// Simple cache of phi matrices to avoid reloading repeatedly across frames
struct PhiCache {
    // key: (coord_char, mol_index) -> matrix
    std::map<std::pair<char,int>, Eigen::MatrixXd> cache;

    const Eigen::MatrixXd& get(char coord, int mol) {
        std::pair<char,int> key{coord, mol};
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;

        std::ostringstream path;
        path << PHI_DIR << "phi" << coord << mol << ".csv";
        Eigen::MatrixXd M = read_csv_matrix(path.str());
        if (M.rows() != mol_beads || M.cols() != mol_beads) {
            std::ostringstream oss;
            oss << "Unexpected phi size for " << path.str()
                << " got " << M.rows() << "x" << M.cols()
                << " expected " << mol_beads << "x" << mol_beads;
            throw std::runtime_error(oss.str());
        }
        auto [it2, ok] = cache.emplace(key, std::move(M));
        return it2->second;
    }
};

// Read entire frame (after headers). We only keep id, xu, yu, zu.
// Sort by id ascending so that molecule blocks are contiguous (1..N).
struct FrameCoords {
    std::vector<double> xu, yu, zu; // size = num_mol * mol_beads
};

FrameCoords read_frame_coords_sorted(const std::string& frame_path) {
    std::ifstream fin(frame_path);
    if (!fin) throw std::runtime_error("Cannot open frame: " + frame_path);
    std::string line;
    for (int i = 0; i < HEADER_LINES; ++i) {
        if (!std::getline(fin, line)) throw std::runtime_error("Unexpected EOF in header: " + frame_path);
    }

    struct Row { int id; double xu, yu, zu; };
    std::vector<Row> rows;
    rows.reserve(num_mol * mol_beads);

    int id, type;
    double x,y,z,xu,yu,zu;
    while (fin >> id >> type >> x >> y >> z >> xu >> yu >> zu) {
        rows.push_back({id, xu, yu, zu});
    }
    if (rows.empty()) throw std::runtime_error("No atom rows in " + frame_path);

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.id < b.id; });

    FrameCoords fc;
    const size_t N = rows.size();
    fc.xu.resize(N);
    fc.yu.resize(N);
    fc.zu.resize(N);
    for (size_t i = 0; i < N; ++i) {
        fc.xu[i] = rows[i].xu;
        fc.yu[i] = rows[i].yu;
        fc.zu[i] = rows[i].zu;
    }
    return fc;
}

// Save Eigen::MatrixXd as CSV
void save_csv_matrix(const std::string& path, const Eigen::MatrixXd& M) {
    std::ofstream fout(path);
    if (!fout) throw std::runtime_error("Cannot write: " + path);
    fout.setf(std::ios::scientific);
    fout << std::setprecision(17);
    for (int r = 0; r < M.rows(); ++r) {
        for (int c = 0; c < M.cols(); ++c) {
            fout << M(r, c);
            if (c + 1 < M.cols()) fout << ",";
        }
        fout << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        // Create output dir once
        if (rank == 0) {
            std::error_code ec;
            if (!fs::exists(OUTPUT_DIR)) fs::create_directories(OUTPUT_DIR, ec);
            if (ec) std::cerr << "Warning: failed to create " << OUTPUT_DIR << ": " << ec.message() << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Discover frames and split among ranks
        auto frames = discover_frames_mpi(rank);
        const int total_frames = static_cast<int>(frames.size());
        if (total_frames == 0) {
            if (rank == 0) std::cerr << "No dump.*.txt in " << INPUT_DIR << "\n";
            MPI_Finalize();
            return 0;
        }
        int per = total_frames / size;
        int rem = total_frames % size;
        int start = rank * per + std::min(rank, rem);
        int end   = start + per + ((rank < rem) ? 1 : 0);

        PhiCache phi_cache; // loads on first use, cached thereafter

        for (int fidx = start; fidx < end; ++fidx) {
            const std::string& frame_path = frames[fidx];
            // read entire frame once
            FrameCoords fc = read_frame_coords_sorted(frame_path);

            // Coeff matrices: rows=mol_beads (modes p), cols=num_mol (molecule index 1..num_mol)
            Eigen::MatrixXd Xp(mol_beads, num_mol);
            Eigen::MatrixXd Yp(mol_beads, num_mol);
            Eigen::MatrixXd Zp(mol_beads, num_mol);

            // for each molecule
            for (int i_mol = 1; i_mol <= num_mol; ++i_mol) {
                const int start_idx = (i_mol - 1) * mol_beads; // 0-based
                // Map slices for this molecule
                Eigen::Map<const Eigen::VectorXd> xu(&fc.xu[start_idx], mol_beads);
                Eigen::Map<const Eigen::VectorXd> yu(&fc.yu[start_idx], mol_beads);
                Eigen::Map<const Eigen::VectorXd> zu(&fc.zu[start_idx], mol_beads);

                // Load phi matrices for this molecule (from CSV)
                const Eigen::MatrixXd& phix = phi_cache.get('x', i_mol); // 720x720
                const Eigen::MatrixXd& phiy = phi_cache.get('y', i_mol);
                const Eigen::MatrixXd& phiz = phi_cache.get('z', i_mol);

                // Project: xp[p] = dot(xu, phix.col(p)), etc.
                for (int p = 0; p < mol_beads; ++p) {
                    Xp(p, i_mol - 1) = xu.dot(phix.col(p));
                    Yp(p, i_mol - 1) = yu.dot(phiy.col(p));
                    Zp(p, i_mol - 1) = zu.dot(phiz.col(p));
                }
            }

            // Build output filenames: prefix + original basename
            const std::string base = fs::path(frame_path).filename().string(); // e.g., dump.0001.txt
            const std::string out_x = OUTPUT_DIR + "x_" + base;
            const std::string out_y = OUTPUT_DIR + "y_" + base;
            const std::string out_z = OUTPUT_DIR + "z_" + base;

            save_csv_matrix(out_x, Xp);
            save_csv_matrix(out_y, Yp);
            save_csv_matrix(out_z, Zp);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cerr << "Done.\n";
    } catch (const std::exception& ex) {
        std::cerr << "[rank " << rank << "] ERROR: " << ex.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}

