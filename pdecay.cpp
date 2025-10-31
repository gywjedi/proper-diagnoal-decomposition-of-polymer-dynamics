// pdecay.cpp
// C++17 + MPI + Eigen translation of the Python pdecay() script.
// Scans base directories for Xp5/x_dump.*.txt, reads row (p-1) from x_/y_/z_ CSVs,
// computes normalized autocorrelation vs dt, writes Pmode<p>.txt.
//
// Output format: "time norm_autocorr" (space-separated, no header).
// time = dt * 100.0  (matches dt*10000*0.01 in the Python)

#include <mpi.h>

#include <algorithm>
#include <cctype>
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

// ---------------- Configuration ----------------
static const std::vector<std::string> BASES = {"./"}; // same as Python's base = ["./"]
static const std::string SUBDIR      = "Xp5";         // where x_/y_/z_ CSVs live
static const int MAX_MODE            = 20;            // pmodes = 1..20
static const double TIME_SCALE       = 100.0;         // dt*10000*0.01 = dt*100
// ------------------------------------------------

// helpers
static inline bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
}
static inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && std::equal(suf.rbegin(), suf.rend(), s.rbegin());
}

// List frame basenames by scanning .../Xp5/x_dump.*.txt and stripping the leading "x_"
std::vector<std::string> list_frame_basenames(const std::string& base) {
    std::vector<std::string> out;
    fs::path dir = fs::path(base) / SUBDIR;
    if (!fs::exists(dir)) {
        throw std::runtime_error("Directory not found: " + dir.string());
    }
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const auto name = e.path().filename().string(); // e.g., x_dump.0001.txt
        if (starts_with(name, "x_dump.") && ends_with(name, ".txt")) {
            // strip leading "x_"
            if (name.size() > 2) {
                out.push_back(name.substr(2)); // "dump.0001.txt"
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Read a single 0-based row from a CSV file with commas, no headers.
// Returns the parsed row as vector<double>. Throws if the row doesn't exist.
std::vector<double> read_csv_row(const std::string& path, int row_index) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open: " + path);
    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        if (line_no == row_index) {
            std::vector<double> row;
            std::istringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                // trim
                auto l = cell.begin();
                while (l != cell.end() && std::isspace(static_cast<unsigned char>(*l))) ++l;
                auto r = cell.end();
                while (r != l && std::isspace(static_cast<unsigned char>(*(r-1)))) --r;
                std::string trimmed(l, r);
                row.push_back(trimmed.empty() ? 0.0 : std::stod(trimmed));
            }
            if (row.empty()) throw std::runtime_error("Empty row in: " + path);
            return row;
        }
        ++line_no;
    }
    std::ostringstream oss;
    oss << "Row " << row_index << " not found in " << path << " (file has " << line_no << " lines)";
    throw std::runtime_error(oss.str());
}

// Convert vector<vector<double>> (rows) to Eigen::MatrixXd (T x N)
Eigen::MatrixXd to_matrix(const std::vector<std::vector<double>>& rows) {
    if (rows.empty()) return Eigen::MatrixXd();
    const int T = static_cast<int>(rows.size());
    const int N = static_cast<int>(rows.front().size());
    Eigen::MatrixXd M(T, N);
    for (int t = 0; t < T; ++t) {
        if (static_cast<int>(rows[t].size()) != N)
            throw std::runtime_error("Inconsistent column count across frames.");
        for (int j = 0; j < N; ++j) M(t, j) = rows[t][j];
    }
    return M;
}

// Compute normalized autocorrelation for a given base dir and mode p (1-based).
// Returns vector of length |dt_range| with values for dt = 1..floor(T/2)-1.
// Also returns dt_range size via out_dt_count (to reuse for "time" construction).
std::vector<double> compute_decay_for_mode(const std::string& base, int p, int& out_dt_count) {
    // 1) Gather frame basenames
    auto basenames = list_frame_basenames(base); // e.g., ["dump.0001.txt", ...]
    if (basenames.empty())
        throw std::runtime_error("No x_dump.*.txt in " + (fs::path(base)/SUBDIR).string());

    // 2) For each frame, load row (p-1) from x_, y_, z_
    std::vector<std::vector<double>> x_rows, y_rows, z_rows;
    x_rows.reserve(basenames.size());
    y_rows.reserve(basenames.size());
    z_rows.reserve(basenames.size());

    for (const auto& fi : basenames) {
        const std::string xfile = (fs::path(base) / SUBDIR / ("x_" + fi)).string();
        const std::string yfile = (fs::path(base) / SUBDIR / ("y_" + fi)).string();
        const std::string zfile = (fs::path(base) / SUBDIR / ("z_" + fi)).string();
        x_rows.push_back(read_csv_row(xfile, p - 1)); // 0-based row
        y_rows.push_back(read_csv_row(yfile, p - 1));
        z_rows.push_back(read_csv_row(zfile, p - 1));
    }

    // 3) Convert to Eigen matrices: T x N
    Eigen::MatrixXd X = to_matrix(x_rows);
    Eigen::MatrixXd Y = to_matrix(y_rows);
    Eigen::MatrixXd Z = to_matrix(z_rows);

    const int T = static_cast<int>(X.rows());
    if (T != static_cast<int>(Y.rows()) || T != static_cast<int>(Z.rows()))
        throw std::runtime_error("Mismatched T across X/Y/Z.");
    const int N = static_cast<int>(X.cols());
    if (N != static_cast<int>(Y.cols()) || N != static_cast<int>(Z.cols()))
        throw std::runtime_error("Mismatched N across X/Y/Z.");

    const int max_dt = T / 2; // Python used int(T/2)
    std::vector<double> rouse_modep;
    rouse_modep.reserve(std::max(0, max_dt - 1));

    for (int dt = 1; dt < max_dt; ++dt) {
        const int rows = T - dt;
        // blocks: [0..T-dt-1] and [dt..T-1]
        auto X0  = X.block(0,   0, rows, N).array();
        auto Xt  = X.block(dt,  0, rows, N).array();
        auto Y0  = Y.block(0,   0, rows, N).array();
        auto Yt  = Y.block(dt,  0, rows, N).array();
        auto Z0  = Z.block(0,   0, rows, N).array();
        auto Zt  = Z.block(dt,  0, rows, N).array();

        const double num = (X0 * Xt).sum() + (Y0 * Yt).sum() + (Z0 * Zt).sum();
        const double den = (X0.square()).sum() + (Y0.square()).sum() + (Z0.square()).sum();

        rouse_modep.push_back(den > 0.0 ? (num / den) : 0.0);
    }

    out_dt_count = static_cast<int>(rouse_modep.size());
    return rouse_modep;
}

// Save two-column (space-separated) file with no header.
void save_time_series(const std::string& outpath,
                      const std::vector<double>& time,
                      const std::vector<double>& values) {
    if (time.size() != values.size())
        throw std::runtime_error("time and values size mismatch.");
    std::ofstream fout(outpath);
    if (!fout) throw std::runtime_error("Cannot write: " + outpath);
    fout.setf(std::ios::scientific);
    fout << std::setprecision(17);
    for (size_t i = 0; i < time.size(); ++i) {
        fout << time[i] << " " << values[i] << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        // Build pmodes = 1..MAX_MODE
        std::vector<int> pmodes;
        pmodes.reserve(MAX_MODE);
        for (int p = 1; p <= MAX_MODE; ++p) pmodes.push_back(p);

        // Split modes across ranks
        const int total_modes = static_cast<int>(pmodes.size());
        const int per = total_modes / size;
        const int rem = total_modes % size;
        const int start = rank * per + std::min(rank, rem);
        const int end   = start + per + ((rank < rem) ? 1 : 0);

        for (int idx = start; idx < end; ++idx) {
            const int p = pmodes[idx];

            // Like the Python, iterate all base dirs; last one wins if multiple.
            // (If you want separate outputs per base, add base-specific suffixes.)
            std::vector<double> last_series;
            int dt_count = 0;

            for (const auto& base : BASES) {
                int dtc = 0;
                auto series = compute_decay_for_mode(base, p, dtc);
                last_series = std::move(series);
                dt_count = dtc;
            }

            // Build time vector: dt = 1..dt_count, t = dt * 100.0
            std::vector<double> time(dt_count);
            for (int i = 0; i < dt_count; ++i) time[i] = (i + 1) * TIME_SCALE;

            std::ostringstream outname;
            outname << "Pmode" << p << ".txt";
            save_time_series(outname.str(), time, last_series);
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

