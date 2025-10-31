// mpi_pca_coords.cpp
// PCA of xu/yu/zu per molecule across dump.*.txt frames.
// Outputs CSV only: phix<M>.csv, phiy<M>.csv, phiz<M>.csv and lamx<M>.csv, lamy<M>.csv, lamz<M>.csv

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace fs = std::filesystem;

// ------------------------- Config -------------------------
static const std::string PATH = "./_coord_/"; // folder with dump.*.txt
static const int HEADER_LINES = 9;            // lines to skip at top of each dump
static const int total_numbers_mol = 100;     // total number of molecules
static const int polymer_weight    = 720;     // atoms per molecule
// ----------------------------------------------------------

enum CoordType { XU, YU, ZU };

static inline bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
}
static inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && std::equal(suf.rbegin(), suf.rend(), s.rbegin());
}

Eigen::VectorXd trapz_weights(int n) {
    Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
    if (n >= 1) w(0) = 0.5;
    if (n >= 2) w(n - 1) = 0.5;
    return w;
}

std::vector<std::string> discover_files_mpi(int rank) {
    std::vector<std::string> files;
    if (rank == 0) {
        if (!fs::exists(PATH))
            throw std::runtime_error("Directory not found: " + PATH);
        for (const auto& e : fs::directory_iterator(PATH)) {
            if (!e.is_regular_file()) continue;
            const auto name = e.path().filename().string();
            if (starts_with(name, "dump.") && ends_with(name, ".txt"))
                files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    }

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

// Read xu/yu/zu for one molecule from a single dump file (after skipping header)
std::vector<double> read_coord_for_molecule(const std::string& filepath,
                                            int start_atid, int end_atid,
                                            CoordType coord) {
    std::ifstream fin(filepath);
    if (!fin) throw std::runtime_error("Failed to open: " + filepath);

    std::string line;
    for (int i = 0; i < HEADER_LINES; ++i) {
        if (!std::getline(fin, line))
            throw std::runtime_error("Unexpected EOF in header: " + filepath);
    }

    struct Row { int atid; double val; };
    std::vector<Row> rows;
    rows.reserve(end_atid - start_atid + 1);

    int atid, atype;
    double x, y, z, xu, yu, zu;
    while (fin >> atid >> atype >> x >> y >> z >> xu >> yu >> zu) {
        if (atid < start_atid || atid > end_atid) continue;
        double value = (coord == XU ? xu : coord == YU ? yu : zu);
        rows.push_back({atid, value});
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.atid < b.atid; });

    std::vector<double> vals;
    vals.reserve(rows.size());
    for (auto& r : rows) vals.push_back(r.val);
    return vals;
}

void save_matrix_csv(const std::string& fname, const Eigen::MatrixXd& M) {
    std::ofstream fout(fname);
    if (!fout) throw std::runtime_error("Cannot write " + fname);
    fout.setf(std::ios::scientific);
    fout << std::setprecision(17);
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            fout << M(i, j);
            if (j + 1 < M.cols()) fout << ",";
        }
        fout << "\n";
    }
}

void save_vector_csv(const std::string& fname, const Eigen::VectorXd& v) {
    std::ofstream fout(fname);
    if (!fout) throw std::runtime_error("Cannot write " + fname);
    fout.setf(std::ios::scientific);
    fout << std::setprecision(17);
    for (int i = 0; i < v.size(); ++i) {
        fout << v(i) << "\n";
    }
}

// PCA for one coordinate, one molecule: build US (ns x nx), center in time, Cov over time, eig, normalize over space
void pca_one_coord_for_molecule(const std::vector<std::string>& fnames,
                                int mol, int nx,
                                CoordType coord,
                                const Eigen::VectorXd& w_time,
                                const Eigen::VectorXd& w_space) {
    const int ns = static_cast<int>(fnames.size());
    const int start_atid = 1 + (mol - 1) * polymer_weight;
    const int end_atid   = mol * polymer_weight;

    // Build US(ns x nx)
    Eigen::MatrixXd US(ns, nx);
    for (int f = 0; f < ns; ++f) {
        auto vals = read_coord_for_molecule(fnames[f], start_atid, end_atid, coord);
        if ((int)vals.size() != nx) {
            std::ostringstream oss;
            oss << "File " << fnames[f] << " returned " << vals.size()
                << " rows for mol " << mol << " (expected " << nx << ")";
            throw std::runtime_error(oss.str());
        }
        for (int j = 0; j < nx; ++j) US(f, j) = vals[j];
    }

    // Mean over space, per time row
    Eigen::VectorXd u_mean(ns);
    for (int i = 0; i < ns; ++i) u_mean(i) = US.row(i).mean();

    // UP(nx x ns): centered columns by time mean
    Eigen::MatrixXd UP(nx, ns);
    for (int i = 0; i < ns; ++i)
        for (int j = 0; j < nx; ++j)
            UP(j, i) = US(i, j) - u_mean(i);

    // Covariance over time with trapezoidal weights
    Eigen::MatrixXd C = UP * w_time.asDiagonal() * UP.transpose();
    C /= static_cast<double>(ns);
    C = 0.5 * (C + C.transpose()); // symmetrize

    // Eigendecomposition (symmetric) -> ascending
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed for mol " + std::to_string(mol));

    // Reverse to descending
    const int n = es.eigenvalues().size();
    Eigen::VectorXd lam_desc(n);
    Eigen::MatrixXd V_desc = es.eigenvectors();
    for (int k = 0; k < n / 2; ++k) {
        V_desc.col(k).swap(V_desc.col(n - 1 - k));
    }
    for (int k = 0; k < n; ++k) lam_desc(k) = es.eigenvalues()(n - 1 - k);

    // Normalize eigenvectors by trapezoidal weights over space
    for (int p = 0; p < n; ++p) {
        double acc = 0.0;
        for (int j = 0; j < nx; ++j) acc += w_space(j) * V_desc(j, p) * V_desc(j, p);
        double norm = std::sqrt(acc);
        if (norm > 0.0) V_desc.col(p) /= norm;
    }

    // Save CSVs
    const char c = (coord == XU ? 'x' : coord == YU ? 'y' : 'z');
    std::ostringstream phiname; phiname << "phi" << c << mol << ".csv";
    std::ostringstream lamname; lamname << "lam" << c << mol << ".csv";

    save_matrix_csv(phiname.str(), V_desc);
    save_vector_csv(lamname.str(), lam_desc);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        auto fnames = discover_files_mpi(rank);
        const int ns = static_cast<int>(fnames.size());
        if (ns == 0) {
            if (rank == 0) std::cerr << "No dump.*.txt files found in " << PATH << "\n";
            MPI_Finalize();
            return 0;
        }

        const int nx = polymer_weight;

        // Distribute molecules across ranks
        int mol_per_node = total_numbers_mol / size;
        int remainder    = total_numbers_mol % size;
        int start_m = rank * mol_per_node + std::min(rank, remainder) + 1;              // inclusive
        int end_m   = start_m + mol_per_node + ((rank < remainder) ? 1 : 0);            // exclusive

        // Precompute trapezoidal weights
        Eigen::VectorXd w_time = trapz_weights(ns);
        Eigen::VectorXd w_space = trapz_weights(nx);

        for (int mol = start_m; mol < end_m; ++mol) {
            pca_one_coord_for_molecule(fnames, mol, nx, XU, w_time, w_space);
            pca_one_coord_for_molecule(fnames, mol, nx, YU, w_time, w_space);
            pca_one_coord_for_molecule(fnames, mol, nx, ZU, w_time, w_space);
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

