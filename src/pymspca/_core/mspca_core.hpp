#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025 Ryan Cory-Wright, Jean Pauphilet (algorithm: msPCA src/msPCA_R_CPP.cpp)
// Copyright (C) 2026 Esa Dhanani (R-independent, header-only port)
//
// Faithful, language-agnostic port of the msPCA iterative deflation heuristic
// (Cory-Wright & Pauphilet, "Sparse PCA With Multiple Components", Operations
// Research 2026). The arithmetic is unchanged; the only differences from the R
// package source are:
//   * Rcpp::List / NumericVector  -> plain structs and std::vector<int>
//   * R::rnorm                    -> injected normal sampler (Context::rnorm)
//   * Rcpp::warning / Rcout       -> injected sinks (Context::warn, Context::log)
//   * the number of outer iterations is reported (n_iter)
// so the same header can back both the R and the Python bindings.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <functional>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <string>
#include <atomic>
#include <thread>
#include <vector>

#include "cov_operator.hpp"

namespace mspca {

namespace constants {  // msPCA src/ConstantArguments.h
constexpr double slowPeriodRate = 0.15;
constexpr double fastPeriodRate = 0.75;
constexpr double changedRateLow = 0.01;
constexpr double changedRateHigh = 0.05;
}  // namespace constants

struct Context {
  std::function<double()> rnorm;                  // standard normal draws
  std::function<void(const std::string&)> warn;   // non-fatal warnings
  std::ostream* log = nullptr;                    // verbose output (nullptr = silent)
};

using MatVec = std::function<Vec(const Vec&)>;

inline Vec truncateVector(const Vec& v, int k) {
  const int n = static_cast<int>(v.size());
  if (k >= n) { Vec y = v; y.normalize(); return y; }
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                   [&](int a, int b) { return std::abs(v[a]) > std::abs(v[b]); });
  Vec y = Vec::Zero(n);
  for (int i = 0; i < k; ++i) y[idx[i]] = v[idx[i]];
  y.normalize();
  return y;
}

inline double evaluateFunctor(const Vec& x, const MatVec& applyM) { return x.dot(applyM(x)); }

// Iterative truncation heuristic (Yuan & Zhang 2013).
inline Vec iterativeTruncHeuristic(int k, const Vec& beta0, const MatVec& applyM) {
  Vec beta = truncateVector(beta0, k);
  for (int i = 0; i < 100; ++i) {
    Vec prev = beta;
    beta = truncateVector(applyM(beta), k);
    if ((beta - prev).squaredNorm() < 1e-12) break;
  }
  return beta;
}

inline void singlePCHeuristic(int k, const MatVec& applyM, int dim, const Vec& beta0,
                              double& lambda_partial, Vec& x_output, Context& ctx,
                              int maxRestartTPM = 10, int timeLimitTPM = 20) {
  Vec bestBeta = iterativeTruncHeuristic(k, beta0, applyM);
  double bestObj = evaluateFunctor(bestBeta, applyM);
  int countdown = maxRestartTPM;
  time_t start = time(0);
  while (countdown > 0 && difftime(time(0), start) < timeLimitTPM) {
    Vec beta(dim);
    for (int i = 0; i < dim; ++i) beta(i) = ctx.rnorm();
    beta.normalize();
    beta = iterativeTruncHeuristic(k, beta, applyM);
    double obj = evaluateFunctor(beta, applyM);
    if (obj > bestObj) { bestObj = obj; bestBeta = beta; countdown = maxRestartTPM; }
    countdown--;
  }
  lambda_partial = bestObj;
  x_output = bestBeta;
}

// ---------------------------------------------------------------------------
// Batched restarts (pymspca extension, opt-in).
// The R package runs each random restart of the truncated power method as its own chain of
// matrix-vector products. Here all restarts of a round are advanced together as one p x b
// block, so each step is a single matrix-matrix product (BLAS-3: far better cache reuse than
// b separate matvecs) followed by column-wise truncation. A round of b restarts that fails
// to improve the incumbent ends the search, mirroring the sequential countdown rule.
using MatMat = std::function<Mat(const Mat&)>;
constexpr int kPanel = 8;  // restarts per panel (BLAS-3 width); fixed so results are thread-count invariant

inline void truncateColumns(Mat& B, int k, std::vector<int>& idx) {
  const int n = static_cast<int>(B.rows());
  for (int j = 0; j < B.cols(); ++j) {
    auto v = B.col(j);
    if (k < n) {
      std::iota(idx.begin(), idx.end(), 0);
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                       [&](int a, int b) { return std::abs(v[a]) > std::abs(v[b]); });
      for (int i = k; i < n; ++i) v[idx[i]] = 0.0;
    }
    v.normalize();
  }
}

// Number of worker threads used by the batched restarts (0 = hardware concurrency).
inline int& numThreads() { static int n = 0; return n; }
inline int resolvedThreads(int work) {
  int n = numThreads() > 0 ? numThreads() : static_cast<int>(std::thread::hardware_concurrency());
  return std::max(1, std::min(n, work));
}

// Advance a block of restarts to convergence: up to 100 truncated power steps per column,
// each column frozen once it stops moving (same rule as iterativeTruncHeuristic).
inline void runBlock(Mat& B, int k, const MatMat& applyMMat) {
  const int b = static_cast<int>(B.cols());
  std::vector<int> idx(B.rows());
  std::vector<char> active(b, 1);
  int nActive = b;
  for (int it = 0; it < 100 && nActive > 0; ++it) {
    Mat Bn = applyMMat(B);
    truncateColumns(Bn, k, idx);
    for (int j = 0; j < b; ++j) {
      if (!active[j]) { Bn.col(j) = B.col(j); continue; }
      if ((Bn.col(j) - B.col(j)).squaredNorm() < 1e-12) { active[j] = 0; --nActive; }
    }
    B.swap(Bn);
  }
}


inline void singlePCHeuristicBatched(int k, const MatVec& applyM, const MatMat& applyMMat, int dim,
                                     const Vec& beta0, double& lambda_partial, Vec& x_output, Context& ctx,
                                     int batch, int timeLimitTPM) {
  Vec bestBeta = iterativeTruncHeuristic(k, beta0, applyM);
  double bestObj = evaluateFunctor(bestBeta, applyM);
  std::vector<int> idx(dim);
  time_t start = time(0);
  while (batch > 0 && difftime(time(0), start) < timeLimitTPM) {
    Mat B(dim, batch);
    for (int j = 0; j < batch; ++j)
      for (int i = 0; i < dim; ++i) B(i, j) = ctx.rnorm();
    truncateColumns(B, k, idx);
    // The block is cut into fixed-width panels whose boundaries depend only on the restart
    // index, never on the thread count, and each panel is advanced on its own. Threads
    // take whole panels. Every floating-point operation is therefore the same for any
    // number of threads, so a given seed gives the same answer on every machine.
    const int nPanels = (batch + kPanel - 1) / kPanel;
    auto runPanel = [&](int q) {
      const int c0 = q * kPanel, w = std::min(kPanel, batch - c0);
      Mat P = B.middleCols(c0, w);
      runBlock(P, k, applyMMat);
      B.middleCols(c0, w) = P;
    };
    const int T = (static_cast<double>(dim) * dim < 1.5e4) ? 1 : resolvedThreads(nPanels);  // tiny problems: threads cost more than they save
    if (T <= 1) {
      for (int q = 0; q < nPanels; ++q) runPanel(q);
    } else {
      std::atomic<int> next(0);
      std::vector<std::thread> pool;
      for (int t = 0; t < T; ++t)
        pool.emplace_back([&] { for (int q; (q = next.fetch_add(1)) < nPanels;) runPanel(q); });
      for (auto& th : pool) th.join();
    }
    const Mat MB = applyMMat(B);
    int jbest = -1;
    double obest = bestObj;
    for (int j = 0; j < batch; ++j) {
      const double o = B.col(j).dot(MB.col(j));
      if (o > obest) { obest = o; jbest = j; }
    }
    if (jbest < 0) break;
    bestObj = obest;
    bestBeta = B.col(jbest);
  }
  lambda_partial = bestObj;
  x_output = bestBeta;
}

inline double fnviolation(const Mat& x, const CovOp& op, int feasibilityConstraintType) {
  const int r = static_cast<int>(x.cols());
  double v = 0;
  Mat y = x.transpose() * x;
  if (feasibilityConstraintType == 0) {
    for (int i = 0; i < r; ++i)
      for (int j = i; j < r; ++j) v += std::fabs(y(i, j) - (i == j ? 1.0 : 0.0));
  } else {
    const double traceSigma = op.trace();
    const double invTraceSigma = (traceSigma > 0.0 ? 1.0 / traceSigma : 1.0);
    Mat C = op.gram(x);
    for (int i = 0; i < r; ++i)
      for (int j = i; j < r; ++j)
        v += (i == j ? std::fabs(y(i, j) - 1.0) : invTraceSigma * std::fabs(C(i, j)));
  }
  return v;
}

struct MultiResult {
  double objective_value;
  double feasibility_violation;
  double runtime;
  Mat x_best;               // p x r, columns sorted by explained variance
  Vec variance_explained;   // r
  double total_variance;
  int n_iter;
};

struct SingleResult {
  double objective_value;
  double runtime;
  Mat x_best;  // p x 1
};

inline MultiResult iterativeDeflationHeuristic(const CovOp& op, int r, std::vector<int> ks,
                                               int feasibilityConstraintType, int maxIter,
                                               double feasibilityTolerance, double stallingTolerance,
                                               int timeLimitTPM, int maxRestartTPM, int minRestartTPM,
                                               Context& ctx, bool batched = false) {
  const int p = op.dim();
  if (static_cast<int>(ks.size()) < r) {
    ctx.warn("You requested " + std::to_string(r) + " PCs but only provided " + std::to_string(ks.size()) +
             " sparsity levels. Ran the algorithm for " + std::to_string(ks.size()) + " PCs instead.");
    r = static_cast<int>(ks.size());
  }
  const double traceSigma = op.trace();
  const double invTraceSigma = (traceSigma > 0 ? 1.0 / traceSigma : 1.0);

  double ofv_best = -1e10;
  double violation_best = p;
  Mat x_best = Mat::Zero(p, r);
  double ofv_secondbest = -1e10;
  Mat x_secondbest = Mat::Zero(p, r);
  Mat x_current = Mat::Zero(p, r);
  double ofv_prev = 0, ofv_overall = 0, theLambda = 0, stepSize = 0;
  Vec weights = Vec::Zero(r);
  const int slowPeriod = static_cast<int>(std::ceil(constants::slowPeriodRate * maxIter));
  const int fastPeriod = static_cast<int>(std::ceil(constants::fastPeriodRate * maxIter));
  int iterDone = 0;

  std::ostream* L = ctx.log;
  if (L) {
    *L << "---- Iterative deflation algorithm for sparse PCA with multiple PCs ---\n"
       << "Dimension: " << p << "\nNumber of PCs: " << r << "\nSparsity pattern: ";
    for (int t = 0; t < r; t++) *L << " " << ks[t];
    *L << "\n" << std::setw(13) << "Iteration |" << std::setw(23) << "Objective value |"
       << std::setw(28) << "Feasibility Violation |" << std::setw(11) << "Time" << "\n";
  }

  auto startTime = std::chrono::high_resolution_clock::now();
  Vec x_output;
  double lambda_partial = 0;

  for (int theIter = 1; theIter <= maxIter; theIter++) {
    iterDone = theIter;
    theLambda += stepSize;
    for (int t = 0; t < r; t++) {
      if (theIter == 1 && ks[t] > p) {
        ctx.warn("For PC #" + std::to_string(t + 1) + ", you requested a sparsity level (" + std::to_string(ks[t]) +
                 ") that exceeds the dimension (" + std::to_string(p) + "). Ran the algorithm with a sparsity level of " +
                 std::to_string(p) + " instead.");
        ks[t] = p;
      }
      const int nOther = r - 1;
      Mat W(p, nOther);
      Vec d(nOther);
      {
        int col = 0;
        for (int s = 0; s < r; s++) {
          if (s != t) {
            if (feasibilityConstraintType == 0) W.col(col) = x_current.col(s);
            else W.col(col) = op.apply(x_current.col(s)) * invTraceSigma;
            d(col) = theLambda * weights[s];
            col++;
          }
        }
      }
      const double psdShift = (nOther > 0)
          ? (d.array() * W.colwise().squaredNorm().transpose().array()).sum() : 0.0;
      auto applyM = [&op, &W, &d, psdShift, nOther](const Vec& beta) -> Vec {
        Vec y = op.apply(beta);
        if (nOther > 0) {
          Vec c = W.transpose() * beta;
          y.noalias() -= W * (d.asDiagonal() * c);
          y.noalias() += psdShift * beta;
        }
        return y;
      };
      Vec beta0;
      if (theIter == 1) { beta0 = Vec::Ones(p); beta0.normalize(); }
      else beta0 = x_current.col(t);

      if (batched) {
        auto applyMMat = [&op, &W, &d, psdShift, nOther](const Mat& B) -> Mat {
          Mat Y = op.applyMat(B);
          if (nOther > 0) {
            Mat Cm = W.transpose() * B;
            Y.noalias() -= W * (d.asDiagonal() * Cm);
            Y.noalias() += psdShift * B;
          }
          return Y;
        };
        singlePCHeuristicBatched(ks[t], applyM, applyMMat, p, beta0, lambda_partial, x_output, ctx,
                                 (theIter == 1 ? maxRestartTPM : minRestartTPM), timeLimitTPM);
      } else {
        singlePCHeuristic(ks[t], applyM, p, beta0, lambda_partial, x_output, ctx,
                          (theIter == 1 ? maxRestartTPM : minRestartTPM), timeLimitTPM);
      }
      x_current.col(t) = x_output;

      if (theIter == 1) {
        Vec Cu = x_output;
        if (feasibilityConstraintType != 0) Cu = op.apply(x_output) * invTraceSigma;
        const double normCu2 = Cu.squaredNorm();
        weights[t] = (normCu2 > 0.0 ? lambda_partial / normCu2 : 0.0);
      }
    }

    ofv_prev = ofv_overall;
    ofv_overall = op.quadForm(x_current);
    double violation = fnviolation(x_current, op, feasibilityConstraintType);
    double violation_forStep = (1e-7 > violation) ? 1e-7 : violation;
    stepSize = (theIter < fastPeriod ? constants::changedRateLow : constants::changedRateHigh) *
               (theIter < slowPeriod ? violation_forStep : ofv_overall / violation_forStep);

    bool stopCriterion = (theIter == maxIter) ||
                         (std::fabs(ofv_prev - ofv_overall) < stallingTolerance && violation < feasibilityTolerance);
    if (L && (maxIter <= 25 || theIter % 10 == 1 || stopCriterion)) {
      double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
      *L << std::setw(11) << theIter << " |" << std::setw(21) << std::setprecision(3) << ofv_overall / traceSigma
         << " |" << std::setw(26) << std::scientific << std::setprecision(2) << violation << " |"
         << std::defaultfloat << std::setw(11) << std::fixed << std::setprecision(3) << secs << std::defaultfloat << "\n";
    }

    if (violation < feasibilityTolerance || (theIter == maxIter && ofv_best < 0)) {
      if (ofv_best < ofv_overall) {
        x_secondbest = x_best; ofv_secondbest = ofv_best;
        x_best = x_current; ofv_best = ofv_overall;
      }
      if (ofv_best > ofv_overall && ofv_overall > ofv_secondbest) {
        x_secondbest = x_current; ofv_secondbest = ofv_overall;
      }
    }
    if (std::fabs(ofv_prev - ofv_overall) < stallingTolerance && violation < feasibilityTolerance) {
      if (ofv_best < 0) { x_best = x_current; ofv_best = ofv_overall; }
      break;
    }
  }

  double runtime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
  violation_best = fnviolation(x_best, op, feasibilityConstraintType);
  if (violation_best > feasibilityTolerance) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Algorithm terminated without finding a feasible solution (within %.3e tolerance). Last iterate has "
                  "a feasibility violation of %.3e. You may want to rerun the algorithm with more iterations (max_iter)",
                  feasibilityTolerance, violation_best);
    ctx.warn(buf);
  }
  ofv_best = op.quadForm(x_best);

  Vec fve(r);
  for (int t = 0; t < r; ++t) fve(t) = x_best.col(t).dot(op.apply(x_best.col(t)));
  std::vector<int> order(r);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return fve(a) > fve(b); });
  Mat x_sorted(p, r);
  Vec fve_sorted(r);
  for (int j = 0; j < r; ++j) { x_sorted.col(j) = x_best.col(order[j]); fve_sorted(j) = fve(order[j]); }

  return MultiResult{ofv_best, violation_best, runtime, x_sorted, fve_sorted, traceSigma, iterDone};
}

inline SingleResult truncatedPowerMethod(const CovOp& op, int k, int maxIter, int timeLimit, Context& ctx) {
  const int p = op.dim();
  if (k > p) {
    ctx.warn("You requested a sparsity level (" + std::to_string(k) + ") that exceeds the dimension (" +
             std::to_string(p) + "). Ran the algorithm with a sparsity level of " + std::to_string(p) + " instead.");
    k = p;
  }
  auto startTime = std::chrono::high_resolution_clock::now();
  Vec x_output;
  double lambda_partial = 0;
  MatVec applyM = [&op](const Vec& v) -> Vec { return op.apply(v); };
  Vec beta0 = op.seed();
  singlePCHeuristic(k, applyM, p, beta0, lambda_partial, x_output, ctx, maxIter, timeLimit);
  double runtime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
  Mat x_best = Mat::Zero(p, 1);
  x_best.col(0) = x_output;
  return SingleResult{lambda_partial, runtime, x_best};
}

}  // namespace mspca
