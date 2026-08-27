// dtfit_lsi.h: fixed-size embedded port of dtfit's streaming LSIFilter.
//
// A header-only float32 specialization of dtfit.streaming.LSIFilter for one
// model at a frozen window/order (see tools/embed_lsi.py, which also emits
// lsi_tables.h and the float64 "golden" this mirrors operation-for-operation).
// No malloc, no recursion, fixed-size stack and state: safe on an MCU hot
// path.
//
// Hot path per sample, once the window is full:
//   beta_data = PROJ * y_window
//   H[:,k]    = legendre_project(t_quad^k)            (model Jacobian)
//   e         = beta_data - H*p                       (spectral innovation)
//   A         = H^T R^-1 H                            (information matrix, N x N)
//   Ppost     = (P^-1 + A)^-1                         (a-posteriori covariance)
//   p        += Ppost * (H^T R^-1 e) ;  P = Ppost + Q
//
// The measurement noise R is diagonal and the state is tiny (N << M), so this
// information (Woodbury) form is used instead of forming/inverting the M x M
// innovation covariance S = H P H^T + R: the only inverses are N x N, which
// removes the dominant M x M inverse from the hot path. It is algebraically
// identical to the covariance form (matches the float64 golden to rounding).
//
// Pure C++; include lsi_tables.h first (or just include this).
#pragma once
#include "lsi_tables.h"

struct LsiFilter {
  float tbuf[LSI_W];
  float ybuf[LSI_W];
  int   count;
  bool  ready;            // an estimate has been produced (window filled)
  float p[LSI_N];         // current parameter estimate
  float P[LSI_N][LSI_N];  // parameter covariance

  // Initialize with an optional initial parameter vector (else zeros).
  void reset(const float* p0 = 0) {
    count = 0;
    ready = false;
    for (int i = 0; i < LSI_N; i++) {
      p[i] = p0 ? p0[i] : 0.0f;
      for (int j = 0; j < LSI_N; j++) P[i][j] = (i == j) ? LSI_P0_DIAG : 0.0f;
    }
  }

  // Model value at the current estimate: y = sum_k p[k] * t^k.
  float predict(float t) const {
    float v = 0.0f, tk = 1.0f;
    for (int k = 0; k < LSI_N; k++) { v += p[k] * tk; tk *= t; }
    return v;
  }

  // In-place N x N inverse by Gauss-Jordan with partial pivoting. N is the
  // parameter count, which is tiny, and this is the only matrix inverse the
  // information form needs.
  static bool invertN(const float A[LSI_N][LSI_N], float out[LSI_N][LSI_N]) {
    float a[LSI_N][2 * LSI_N];
    for (int i = 0; i < LSI_N; i++)
      for (int j = 0; j < LSI_N; j++) {
        a[i][j] = A[i][j];
        a[i][j + LSI_N] = (i == j) ? 1.0f : 0.0f;
      }
    for (int c = 0; c < LSI_N; c++) {
      int piv = c;
      float best = a[c][c] < 0 ? -a[c][c] : a[c][c];
      for (int r = c + 1; r < LSI_N; r++) {
        float v = a[r][c] < 0 ? -a[r][c] : a[r][c];
        if (v > best) { best = v; piv = r; }
      }
      if (best < 1e-20f) return false;
      if (piv != c)
        for (int j = 0; j < 2 * LSI_N; j++) {
          float tmp = a[c][j]; a[c][j] = a[piv][j]; a[piv][j] = tmp;
        }
      float d = a[c][c];
      for (int j = 0; j < 2 * LSI_N; j++) a[c][j] /= d;
      for (int r = 0; r < LSI_N; r++) {
        if (r == c) continue;
        float f = a[r][c];
        for (int j = 0; j < 2 * LSI_N; j++) a[r][j] -= f * a[c][j];
      }
    }
    for (int i = 0; i < LSI_N; i++)
      for (int j = 0; j < LSI_N; j++) out[i][j] = a[i][j + LSI_N];
    return true;
  }

  // Ingest one (t, y) sample; returns true if the estimate was updated.
  bool update(float t, float y) {
    if (count < LSI_W) {
      tbuf[count] = t; ybuf[count] = y; count++;
    } else {
      for (int i = 1; i < LSI_W; i++) { tbuf[i - 1] = tbuf[i]; ybuf[i - 1] = ybuf[i]; }
      tbuf[LSI_W - 1] = t; ybuf[LSI_W - 1] = y;
    }
    if (count < LSI_W) return false;

    // beta_data = PROJ * y_window
    float beta_data[LSI_M];
    for (int j = 0; j < LSI_M; j++) {
      float s = 0.0f;
      for (int w = 0; w < LSI_W; w++) s += LSI_PROJ[j][w] * ybuf[w];
      beta_data[j] = s;
    }

    // model Jacobian H[j][k] = NORM[j] * sum_i QW[i] * t_quad[i]^k * LEGV[i][j]
    const float t0 = tbuf[0], tn = tbuf[LSI_W - 1];
    float H[LSI_M][LSI_N];
    for (int j = 0; j < LSI_M; j++)
      for (int k = 0; k < LSI_N; k++) H[j][k] = 0.0f;
    for (int i = 0; i < LSI_QN; i++) {
      float tq = t0 + (tn - t0) * (LSI_QNODES[i] + 1.0f) * 0.5f;
      float wq = LSI_QW[i];
      float bk = 1.0f;  // t_quad^k
      for (int k = 0; k < LSI_N; k++) {
        float wb = wq * bk;
        for (int j = 0; j < LSI_M; j++) H[j][k] += wb * LSI_LEGV[i][j];
        bk *= tq;
      }
    }
    for (int j = 0; j < LSI_M; j++)
      for (int k = 0; k < LSI_N; k++) H[j][k] *= LSI_NORM[j];

    // innovation e = beta_data - H*p
    float e[LSI_M];
    for (int j = 0; j < LSI_M; j++) {
      float bm = 0.0f;
      for (int k = 0; k < LSI_N; k++) bm += H[j][k] * p[k];
      e[j] = beta_data[j] - bm;
    }

    // Information-form measurement update, R diagonal and N << M, so every
    // inverse below is N x N:
    //   A = H^T R^-1 H  (N x N) ,  b = H^T R^-1 e  (N)
    float Rinv[LSI_M];
    for (int a = 0; a < LSI_M; a++) Rinv[a] = 1.0f / LSI_RDIAG[a];
    float A[LSI_N][LSI_N];
    float b[LSI_N];
    for (int i = 0; i < LSI_N; i++) {
      float bi = 0.0f;
      for (int a = 0; a < LSI_M; a++) bi += H[a][i] * Rinv[a] * e[a];
      b[i] = bi;
      for (int j = 0; j < LSI_N; j++) {
        float s = 0.0f;
        for (int a = 0; a < LSI_M; a++) s += H[a][i] * Rinv[a] * H[a][j];
        A[i][j] = s;
      }
    }

    // P_post = (P^-1 + A)^-1, the a-posteriori covariance, before adding Q
    float Pinv[LSI_N][LSI_N], Msum[LSI_N][LSI_N], Ppost[LSI_N][LSI_N];
    if (!invertN(P, Pinv)) return false;
    for (int i = 0; i < LSI_N; i++)
      for (int j = 0; j < LSI_N; j++) Msum[i][j] = Pinv[i][j] + A[i][j];
    if (!invertN(Msum, Ppost)) return false;

    // p += P_post b
    for (int i = 0; i < LSI_N; i++) {
      float s = 0.0f;
      for (int j = 0; j < LSI_N; j++) s += Ppost[i][j] * b[j];
      p[i] += s;
    }

    // P = P_post + Q
    for (int i = 0; i < LSI_N; i++)
      for (int j = 0; j < LSI_N; j++)
        P[i][j] = Ppost[i][j] + (i == j ? LSI_QDIAG[i] : 0.0f);

    ready = true;
    return true;
  }
};
