// mpi_pca_xu.cpp
// C++17 + MPI rewrite of the provided Python script.
// Reads ./_coord_/dump.*.txt, extracts xu for each molecule across files,
// builds covariance via trapezoidal rule over time, eigendecomposes, and
// writes eigenvalues (lamx.txt) and eigenvectors (phix<mol>.(mat|csv)).

#include <mpi.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#ifdef USE_MATIO
#include <matio.h>
#endif

namespace fs = std::filesystem;

static const std::string PATH = "./_coord_/";
static const int HEADER_LINES = 9;

// Helper: string starts/ends with
bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}
bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

// Trapezoidal rule for y with constant spacing dx=1
double trapz_unit_dx(const std::vector<double>& y) {
    if (y.empty()) return 0.0;
    if (y.size() == 1) return 0.0; // single point integral ~ 0 (matches numpy.trapz default behavior)
    double sum = 0.0;
    // sum of interior points
    for (size_t i = 1; i + 1 < y.size(); ++i) sum += y[i];
    // add half endpoints
    sum += 0.5 * (y.front() + y.back());
    return sum; // dx = 1
}

// Build trapezoidal weights (unit spacing) for length n
Eigen::VectorXd trapz_weights(int n) {
    Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
    if (n >= 1) w(0) = 0.5;
    if (n >= 2) w(n - 1) = 0.5;
    return w;
}

// Read xu for a molecule from a single dump file.
// Expected columns after skipping header: atid atype x y z xu yu zu
// Keep rows with atid in [start_atid, end_atid] (inclusive), sorted by atid.
std::vector<double> read_xu_for_molecule(const std::string& filepath,
                                         int start_atid, int end_atid) {
    std::ifstream fin(filepath);
    if (!fin) {
        throw std::runtime_error("Failed to open: " + filepath);
    }

    std::string line;
    for (int i = 0; i < HEADER_LINES; ++i) {
        if (!std::getline(fin, line)) {
            throw std::runtime_error("Unexpected EOF in header: " + filepath);
        }
    }

    struct Row { int atid; double xu; };
    std::vector<Row> rows;
    rows.reserve(end_atid - start_atid + 1);

    // Read whitespace-separated tokens
    // cols: atid (int) atype x y z xu yu zu
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int atid, atype;
        double x, y, z, xu, yu, zu;
        if (!(iss >> atid >> atype >> x >> y >> z >> xu >> yu >> zu)) {
            // skip malformed lines silently
            continue;
        }
        if (atid >= start_atid && atid <= end_atid) {
            rows.push_back({atid, xu});
        }
    }

    // Sort by atid to match Python behavior
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.atid < b.atid; });

    std::vector<double> xu_vals;
    xu_vals.reserve(rows.size());
    for (const auto& r : rows) xu_vals.push_back(r.xu);
    return xu_vals;
}

// Save eigenvalues; to mirror Python behavior this writes lamx.txt per molecule (last write wins).
void save_lambda_txt(const Eigen::VectorXd& lam, const std::string& fname = "lamx.txt") {
    std::ofstream fout(fname);
    if (!fout) throw std::runtime_error("Cannot write " + fname);
    for (int i = 0; i < lam.size(); ++i) {
        fout << std::setprecision(17) << lam(i) << "\n";
    }
}

// Save phi (nx x nx) to MAT (if available) or CSV fallback.
void save_phi(int mol, const Eigen::MatrixXd& phi) {
#ifdef USE_MATIO
    std::string fname = "phix" + std::to_string(mol) + ".mat";
    mat_t* matfp = Mat_CreateVer(fname.c_str(), nullptr, MAT_FT_DEFAULT);
    if (!matfp) throw std::runtime_error("Cannot create MAT file: " + fname);

    size_t dims[2];
    dims[0] = static_cast<size_t>(phi.rows());
    dims[1] = static_cast<size_t>(phi.cols());

    // MATIO expects column-major double array (Eigen is column-major by default)
    matvar_t* matvar = Mat_VarCreate("phi", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims,
                                     const_cast<double*>(phi.data()), 0);
    if (!matvar) {
        Mat_Close(matfp);
        throw std::runtime_error("Mat_VarCreate failed for " + fname);
    }
    if (Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE) != 0) {
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        throw std::runtime_error("Mat_VarWrite failed for " + fname);
    }
    Mat_VarFree(matvar);
    Mat_Close(matfp);
#else
    std::string fname = "phix" + std::to_string(mol) + ".csv";
    std::ofstream fout(fname);
    if (!fout) throw std::runtime_error("Cannot write " + fname);
    fout.setf(std::ios::scientific);
    fout << std::setprecision(17);
    for (int i = 0; i < phi.rows(); ++i) {
        for (int j = 0; j < phi.cols(); ++j) {
            fout << phi(i, j);
            if (j + 1 < phi.cols()) fout << ",";
        }
        fout << "\n";
    }
#endif
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        // 1) Discover files: PATH/dump.*.txt
        std::vector<std::string> fnames;
        if (rank == 0) {
            if (!fs::exists(PATH)) {
                throw std::runtime_error("Directory not found: " + PATH);
            }
            for (const auto& entry : fs::directory_iterator(PATH)) {
                if (!entry.is_regular_file()) continue;
                const std::string name = entry.path().filename().string();
                if (starts_with(name, "dump.") && ends_with(name, ".txt")) {
                    fnames.emplace_back(entry.path().string());
                }
            }
            std::sort(fnames.begin(), fnames.end());
        }

        // Broadcast file list
        // First, broadcast size
        int nf = static_cast<int>(fnames.size());
        MPI_Bcast(&nf, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) fnames.resize(nf);
        // Then broadcast each filename length + content
        for (int i = 0; i < nf; ++i) {
            int len = (rank == 0) ? static_cast<int>(fnames[i].size()) : 0;
            MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0) fnames[i].resize(len);
            MPI_Bcast(fnames[i].data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
        }

        // 2) Parameters (match Python)
        const int total_numbers_mol = 100;
        const int polymer_weight    = 720;

        // Distribute molecules across ranks (same as Python)
        int mol_per_node = total_numbers_mol / size;
        int remainder    = total_numbers_mol % size;
        int start_f = rank * mol_per_node + std::min(rank, remainder) + 1; // inclusive
        int end_f   = start_f + mol_per_node + ((rank < remainder) ? 1 : 0); // exclusive

        // Precompute trapezoidal weights
        const int ns = nf;                 // number of files (time samples)
        const int nx = polymer_weight;     // atoms per molecule (spatial dof)
        Eigen::VectorXd w_time = trapz_weights(ns);
        Eigen::VectorXd w_space = trapz_weights(nx);

        for (int mol = start_f; mol < end_f; ++mol) {
            const int start_atid = 1 + (mol - 1) * polymer_weight;
            const int end_atid   = mol * polymer_weight;

            // Build US matrix: ns x nx (rows: files/time; cols: atoms in molecule ordered by atid)
            Eigen::MatrixXd US(ns, nx);
            for (int f = 0; f < ns; ++f) {
                const std::string& file = fnames[f];
                auto xu = read_xu_for_molecule(file, start_atid, end_atid);

                if (static_cast<int>(xu.size()) != nx) {
                    std::ostringstream oss;
                    oss << "File " << file << " returned " << xu.size()
                        << " rows for mol " << mol << " (expected " << nx << ")";
                    throw std::runtime_error(oss.str());
                }
                for (int j = 0; j < nx; ++j) {
                    US(f, j) = xu[j];
                }
            }

            // u_mean over space (per file): same as Python u_mean[i] = mean_j US[i,j]
            Eigen::VectorXd u_mean(ns);
            for (int i = 0; i < ns; ++i) {
                u_mean(i) = US.row(i).mean();
            }

            // up(j,i) = US(i,j) - u_mean(i); We'll form UP = (nx x ns)
            Eigen::MatrixXd UP(nx, ns);
            for (int i = 0; i < ns; ++i) {
                for (int j = 0; j < nx; ++j) {
                    UP(j, i) = US(i, j) - u_mean(i);
                }
            }

            // Covariance via trapezoid over time (dx=1), then / ns (match Python's /ns)
            // C = UP * diag(w_time) * UP^T  ; then / ns
            Eigen::MatrixXd Wt = w_time.asDiagonal();
            Eigen::MatrixXd C = (UP * Wt * UP.transpose()) / static_cast<double>(ns);

            // Symmetrize lightly in case of roundoff
            C = 0.5 * (C + C.transpose());

            // Eigen decomposition (symmetric)
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
            if (es.info() != Eigen::Success) {
                throw std::runtime_error("Eigen decomposition failed for mol " + std::to_string(mol));
            }
            Eigen::VectorXd lam = es.eigenvalues();      // ascending
            Eigen::MatrixXd V   = es.eigenvectors();     // columns = eigenvectors

            // Sort descending to match Python
            Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(nx, nx - 1, 0); // reverse
            Eigen::VectorXd lam_desc(nx);
            Eigen::MatrixXd V_desc(nx, nx);
            for (int k = 0; k < nx; ++k) {
                lam_desc(k) = lam(idx(k));
                V_desc.col(k) = V.col(idx(k));
            }

            // phi = V_desc; Normalize each column with trapezoid weights over space
            Eigen::MatrixXd phi = V_desc;
            for (int p = 0; p < nx; ++p) {
                // norm = sqrt(trapz(phi(:,p).^2, dx=1))
                double acc = 0.0;
                // weighted sum: w_space(j) * phi(j,p)^2, but remember trapezoid formula sums
                // weights exactly as implemented in trapz; equivalently:
                // trapz = sum(phi^2) - 0.5*(endpoints); with dx=1
                // Here we just use weights vector where ends are 0.5
                for (int j = 0; j < nx; ++j) acc += w_space(j) * phi(j, p) * phi(j, p);
                double norm = std::sqrt(acc);
                if (norm > 0.0) phi.col(p) /= norm;
            }

            // Save outputs (note: lamx.txt will be overwritten by last molecule processed on this rank,
            // matching Python's behavior). If you want per-molecule eigenvalues, change the filename here.
            if (rank == 0) {
                // To avoid parallel races on lamx.txt, only rank 0 writes it.
                // (Python could overwrite across ranks; this is a mild safety improvement.)
                save_lambda_txt(lam_desc, "lamx.txt");
            }
            // Save phi per molecule (each rank writes its own molecule file)
            save_phi(mol, phi);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cerr << "Done.\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "[rank " << rank << "] ERROR: " << ex.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}

