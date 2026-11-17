// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Esa Dhanani
//
// pybind11 bindings for the msPCA core. Heavy lifting runs with the GIL released
// unless verbose output is requested.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>

#include "mspca_core.hpp"

namespace py = pybind11;
using namespace mspca;

namespace {

// Portable standard-normal sampler: std::mt19937_64 is bit-identical across
// standard libraries (std::normal_distribution is not), so we pair it with an
// explicit Marsaglia polar transform. Same seed -> same draws on every platform.
class Normal {
 public:
  explicit Normal(uint64_t seed) : gen_(seed) {}
  double operator()() {
    if (has_spare_) { has_spare_ = false; return spare_; }
    double u, v, s;
    do {
      u = 2.0 * unif() - 1.0;
      v = 2.0 * unif() - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    const double m = std::sqrt(-2.0 * std::log(s) / s);
    spare_ = v * m;
    has_spare_ = true;
    return u * m;
  }
 private:
  double unif() { return (gen_() >> 11) * (1.0 / 9007199254740992.0); }  // 53-bit uniform in [0,1)
  std::mt19937_64 gen_;
  double spare_ = 0.0;
  bool has_spare_ = false;
};

// Streams verbose output to Python's sys.stdout line by line (GIL is held when verbose).
class PyLineBuf : public std::stringbuf {
 public:
  int sync() override {
    py::module_::import("sys").attr("stdout").attr("write")(str());
    py::module_::import("sys").attr("stdout").attr("flush")();
    str("");
    return 0;
  }
};

struct Runner {
  Normal normal;
  std::vector<std::string> warnings;
  PyLineBuf buf;
  std::ostream out{&buf};
  Context ctx;
  bool verbose;

  Runner(uint64_t seed, bool verbose_) : normal(seed), verbose(verbose_) {
    ctx.rnorm = [this]() { return normal(); };
    ctx.warn = [this](const std::string& w) { warnings.push_back(w); };
    ctx.log = verbose ? &out : nullptr;
  }
  void finish() { if (verbose) out.flush(); }
};

py::dict pack(const MultiResult& r, const std::vector<std::string>& w) {
  py::dict d;
  d["objective_value"] = r.objective_value;
  d["feasibility_violation"] = r.feasibility_violation;
  d["runtime"] = r.runtime;
  d["x_best"] = r.x_best;
  d["variance_explained"] = r.variance_explained;
  d["total_variance"] = r.total_variance;
  d["n_iter"] = r.n_iter;
  d["warnings"] = w;
  return d;
}

py::dict pack(const SingleResult& r, const std::vector<std::string>& w) {
  py::dict d;
  d["objective_value"] = r.objective_value;
  d["runtime"] = r.runtime;
  d["x_best"] = r.x_best;
  d["warnings"] = w;
  return d;
}

template <class F>
auto run(bool verbose, F&& f) {
  if (verbose) return f();
  py::gil_scoped_release release;
  return f();
}

py::dict multi(const CovOp& op, int r, std::vector<int> ks, int ctype, bool verbose, int maxIter,
               double feasTol, double stallTol, int timeLimitTPM, int maxRestart, int minRestart,
               uint64_t seed, bool batched) {
  Runner R(seed, verbose);
  MultiResult res = run(verbose, [&] {
    return iterativeDeflationHeuristic(op, r, ks, ctype, maxIter, feasTol, stallTol, timeLimitTPM, maxRestart,
                                       minRestart, R.ctx, batched);
  });
  R.finish();
  return pack(res, R.warnings);
}

py::dict single(const CovOp& op, int k, int maxIter, bool verbose, int timeLimit, uint64_t seed) {
  Runner R(seed, verbose);
  SingleResult res = run(verbose, [&] { return truncatedPowerMethod(op, k, maxIter, timeLimit, R.ctx); });
  R.finish();
  return pack(res, R.warnings);
}

using FMat = Eigen::Ref<const Eigen::MatrixXd>;  // zero-copy for Fortran-ordered float64
using IVec = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using DVec = py::array_t<double, py::array::c_style | py::array::forcecast>;

SpMap csc_map(int nrows, int ncols, const DVec& data, const IVec& indices, const IVec& indptr) {
  if (indptr.size() != ncols + 1) throw std::invalid_argument("indptr must have length ncols + 1");
  return SpMap(nrows, ncols, static_cast<Eigen::Index>(data.size()), indptr.data(), indices.data(), data.data());
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ core of pymspca (port of the msPCA R package by Cory-Wright & Pauphilet)";

  m.def("set_num_threads", [](int n) { numThreads() = n; }, "Threads for batched restarts (0 = all cores)");
  m.def("get_num_threads", []() { return resolvedThreads(1 << 30); });

  m.def("mspca_sigma",
        [](FMat Sigma, int r, std::vector<int> ks, int ctype, bool verbose, int maxIter, double feasTol,
           double stallTol, int timeLimitTPM, int maxRestart, int minRestart, uint64_t seed, bool batched) {
          DenseOp op(Sigma);
          return multi(op, r, ks, ctype, verbose, maxIter, feasTol, stallTol, timeLimitTPM, maxRestart, minRestart, seed, batched);
        });

  m.def("mspca_x",
        [](FMat X, double invDivisor, int r, std::vector<int> ks, int ctype, bool verbose, int maxIter,
           double feasTol, double stallTol, int timeLimitTPM, int maxRestart, int minRestart, uint64_t seed, bool batched) {
          GramOp op(X, invDivisor);
          return multi(op, r, ks, ctype, verbose, maxIter, feasTol, stallTol, timeLimitTPM, maxRestart, minRestart, seed, batched);
        });

  m.def("mspca_sparse",
        [](int nrows, int ncols, DVec data, IVec indices, IVec indptr, DVec mu, DVec invSd, double invDivisor, int r,
           std::vector<int> ks, int ctype, bool verbose, int maxIter, double feasTol, double stallTol,
           int timeLimitTPM, int maxRestart, int minRestart, uint64_t seed, bool batched) {
          SpMap X = csc_map(nrows, ncols, data, indices, indptr);
          Eigen::Map<const Eigen::VectorXd> muv(mu.data(), mu.size()), sdv(invSd.data(), invSd.size());
          SparseGramOp op(X, muv, sdv, invDivisor);
          return multi(op, r, ks, ctype, verbose, maxIter, feasTol, stallTol, timeLimitTPM, maxRestart, minRestart, seed, batched);
        });

  m.def("tpm_sigma", [](FMat Sigma, int k, int maxIter, bool verbose, int timeLimit, uint64_t seed) {
    DenseOp op(Sigma);
    return single(op, k, maxIter, verbose, timeLimit, seed);
  });

  m.def("tpm_x", [](FMat X, double invDivisor, int k, int maxIter, bool verbose, int timeLimit, uint64_t seed) {
    GramOp op(X, invDivisor);
    return single(op, k, maxIter, verbose, timeLimit, seed);
  });

  m.def("tpm_sparse", [](int nrows, int ncols, DVec data, IVec indices, IVec indptr, DVec mu, DVec invSd,
                         double invDivisor, int k, int maxIter, bool verbose, int timeLimit, uint64_t seed) {
    SpMap X = csc_map(nrows, ncols, data, indices, indptr);
    Eigen::Map<const Eigen::VectorXd> muv(mu.data(), mu.size()), sdv(invSd.data(), invSd.size());
    SparseGramOp op(X, muv, sdv, invDivisor);
    return single(op, k, maxIter, verbose, timeLimit, seed);
  });
}
