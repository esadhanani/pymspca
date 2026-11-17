#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025 Ryan Cory-Wright, Jean Pauphilet (DenseOp, GramOp: from msPCA src/CovOperator.h)
// Copyright (C) 2026 Esa Dhanani (R-independent port, Eigen::Ref views, SparseGramOp)
//
// Covariance-operator abstraction. The sparse-PCA core accesses Sigma only through
// a matvec, the trace, the quadratic form tr(x^T Sigma x), the small Gram x^T Sigma x
// and a leading-eigenvector seed. DenseOp and GramOp are line-for-line ports of the
// msPCA R package. SparseGramOp is new: it applies the covariance of a *sparse* data
// matrix with implicit centering/scaling, so centering never densifies X.

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace mspca {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using MatRef = Eigen::Ref<const Eigen::MatrixXd>;
using VecRef = Eigen::Ref<const Eigen::VectorXd>;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMap = Eigen::Map<const SpMat>;

struct CovOp {
  virtual int dim() const = 0;
  virtual Vec apply(const Vec& v) const = 0;
  virtual double trace() const = 0;
  virtual double quadForm(const Mat& x) const = 0;
  virtual Mat gram(const Mat& x) const = 0;
  virtual Vec seed() const = 0;
  // Sigma * B for a p x b block (BLAS-3 path used by the batched restarts).
  virtual Mat applyMat(const Mat& B) const {
    Mat out(B.rows(), B.cols());
    for (int j = 0; j < B.cols(); ++j) out.col(j) = apply(B.col(j));
    return out;
  }
  virtual ~CovOp() = default;
};

// Power iteration seed shared by the implicit operators (identical to msPCA GramOp::seed).
inline Vec powerSeed(const CovOp& op) {
  const int p = op.dim();
  Vec b = Vec::Ones(p);
  b.normalize();
  for (int i = 0; i < 100; ++i) {
    Vec nb = op.apply(b);
    double nm = nb.norm();
    if (nm < 1e-12) break;
    Vec next = nb / nm;
    if ((next - b).squaredNorm() < 1e-16) { b = next; break; }
    b = next;
  }
  return b;
}

// Dense p x p covariance/correlation matrix.
struct DenseOp : public CovOp {
  MatRef S;
  explicit DenseOp(const MatRef& S_) : S(S_) {}
  int dim() const override { return static_cast<int>(S.rows()); }
  Vec apply(const Vec& v) const override { return S * v; }
  double trace() const override { return S.trace(); }
  Mat applyMat(const Mat& B) const override { return S * B; }
  double quadForm(const Mat& x) const override { return (x.transpose() * S * x).trace(); }
  Mat gram(const Mat& x) const override { return x.transpose() * S * x; }
  Vec seed() const override {
    Eigen::SelfAdjointEigenSolver<Mat> solver(S);
    int index;
    solver.eigenvalues().maxCoeff(&index);
    return solver.eigenvectors().col(index);
  }
};

// Dense, already centered/scaled n x p data matrix: Sigma = s * X^T X, never formed.
// X is held as an owning Eigen::MatrixXd (one O(np) copy), not an Eigen::Ref view: with a
// runtime-strided Ref, Eigen dispatches X * x (n x r GEMM) to a different kernel whose
// summation order differs in the last bit, and the iterative heuristic amplifies that into
// a different local optimum. Owning the matrix keeps results bit-identical to the R package.
struct GramOp : public CovOp {
  Mat X;
  double s;
  GramOp(const MatRef& X_, double s_) : X(X_), s(s_) {}
  int dim() const override { return static_cast<int>(X.cols()); }
  Vec apply(const Vec& v) const override {
    const Vec Xv = X * v;  // keep X^T (X v) parenthesised: never build X^T X
    Vec out(X.cols());
    out.noalias() = X.transpose() * Xv;
    out *= s;
    return out;
  }
  double trace() const override { return X.squaredNorm() * s; }
  Mat applyMat(const Mat& B) const override {
    const Mat XB = X * B;
    Mat out(X.cols(), B.cols());
    out.noalias() = X.transpose() * XB;
    out *= s;
    return out;
  }
  double quadForm(const Mat& x) const override { const Mat Xx = X * x; return Xx.squaredNorm() * s; }
  Mat gram(const Mat& x) const override {
    const Mat Xx = X * x;
    Mat G(x.cols(), x.cols());
    G.noalias() = Xx.transpose() * Xx;
    G *= s;
    return G;
  }
  Vec seed() const override { return powerSeed(*this); }
};

// NEW: sparse n x p data matrix X (CSC), with implicit centering and scaling.
//   Xproc = (X - 1 mu^T) D^{-1},  D = diag(sd)   (mu = 0 / sd = 1 when disabled)
//   Sigma v = s * Xproc^T Xproc v
//           = s * D^{-1} [ X^T (X w - 1 (mu^T w)) - mu * 1^T (X w - 1 (mu^T w)) ],  w = D^{-1} v
// Cost per matvec O(nnz(X) + n + p): centering a sparse matrix explicitly would make it
// dense (O(np) memory); here it is a rank-one correction applied on the fly.
struct SparseGramOp : public CovOp {
  SpMap X;
  VecRef mu;     // column means (zeros if center = false)
  VecRef invSd;  // 1 / column scales (ones if scale = false)
  double s;
  double traceCache;

  SparseGramOp(const SpMap& X_, const VecRef& mu_, const VecRef& invSd_, double s_)
      : X(X_), mu(mu_), invSd(invSd_), s(s_) {
    // ||Xproc_j||^2 = (sum_{nonzeros} (x - mu_j)^2 + (n - nnz_j) mu_j^2) / sd_j^2.
    // Two-pass form, as in the R patch: the one-pass ||X_j||^2 - 2 mu_j sum X_ij + n mu_j^2
    // cancels catastrophically when |mu_j| >> sd_j.
    const double n = static_cast<double>(X.rows());
    double tr = 0.0;
    for (int j = 0; j < X.outerSize(); ++j) {
      double c = 0.0, nnz = 0.0;
      for (SpMap::InnerIterator it(X, j); it; ++it) { const double d = it.value() - mu[j]; c += d * d; nnz += 1.0; }
      c += (n - nnz) * mu[j] * mu[j];
      tr += c * invSd[j] * invSd[j];
    }
    traceCache = tr * s;
  }

  int dim() const override { return static_cast<int>(X.cols()); }

  // u = Xproc v  (n-vector)
  Vec forward(const Vec& v) const {
    const Vec w = invSd.cwiseProduct(v);
    Vec u = X * w;
    u.array() -= mu.dot(w);
    return u;
  }
  // Xproc^T u  (p-vector)
  Vec backward(const Vec& u) const {
    Vec out = X.transpose() * u;
    out -= mu * u.sum();
    return invSd.cwiseProduct(out);
  }

  Vec apply(const Vec& v) const override { Vec out = backward(forward(v)); out *= s; return out; }
  Mat applyMat(const Mat& B) const override {
    const Mat Wm = invSd.asDiagonal() * B;
    Mat U = X * Wm;
    U.rowwise() -= (mu.transpose() * Wm);
    Mat out = X.transpose() * U;
    out -= mu * U.colwise().sum();
    out = invSd.asDiagonal() * out;
    out *= s;
    return out;
  }
  double trace() const override { return traceCache; }
  Mat project(const Mat& x) const {
    Mat Xx(X.rows(), x.cols());
    for (int j = 0; j < x.cols(); ++j) Xx.col(j) = forward(x.col(j));
    return Xx;
  }
  double quadForm(const Mat& x) const override { return project(x).squaredNorm() * s; }
  Mat gram(const Mat& x) const override {
    const Mat Xx = project(x);
    Mat G(x.cols(), x.cols());
    G.noalias() = Xx.transpose() * Xx;
    G *= s;
    return G;
  }
  Vec seed() const override { return powerSeed(*this); }
};

}  // namespace mspca
