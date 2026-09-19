/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.feature.onlinekfmc;

import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.util.Preconditions;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.NormOps_DDRM;

/**
 * Small dense linear-algebra and poly-kernel helpers used by {@link OnlineKFMC}. All matrices are
 * {@link DenseMatrix} (column-major). Sizes involved (rank {@code r}, feature dim {@code m}) are
 * expected to be small; matmul/invert/transpose/spectralNorm delegate to EJML, everything else
 * (kernel evaluation, elementwise ops) stays as plain loops since EJML has no equivalent.
 */
class KFMCMath {

    private KFMCMath() {}

    /** Wraps a column-major {@link DenseMatrix} as an EJML matrix without a manual transpose. */
    private static DMatrixRMaj toEjml(DenseMatrix A) {
        return new DMatrixRMaj(A.numRows(), A.numCols(), false, A.values);
    }

    private static DenseMatrix toDense(DMatrixRMaj A) {
        DenseMatrix result = new DenseMatrix(A.numRows, A.numCols);
        for (int i = 0; i < A.numRows; i++) {
            for (int j = 0; j < A.numCols; j++) {
                result.values[i + A.numRows * j] = A.get(i, j);
            }
        }
        return result;
    }

    /** {@code (dot + c) ^ q}. */
    static double poly(double dot, double c, double q) {
        return Math.pow(dot + c, q);
    }

    /** Column {@code col} of {@code D} dotted with {@code x} (both length {@code D.numRows()}). */
    private static double dotColumn(DenseMatrix D, int col, double[] x) {
        int m = D.numRows();
        double s = 0;
        for (int i = 0; i < m; i++) {
            s += D.values[i + m * col] * x[i];
        }
        return s;
    }

    /** {@code kernel(D, D)}: r x r matrix with entries {@code (D_i . D_j + c) ^ q}. */
    static DenseMatrix kernelDD(DenseMatrix D, double c, double q) {
        int m = D.numRows();
        int r = D.numCols();
        DenseMatrix result = new DenseMatrix(r, r);
        for (int i = 0; i < r; i++) {
            for (int j = i; j < r; j++) {
                double dot = 0;
                for (int k = 0; k < m; k++) {
                    dot += D.values[k + m * i] * D.values[k + m * j];
                }
                double v = poly(dot, c, q);
                result.values[i + r * j] = v;
                result.values[j + r * i] = v;
            }
        }
        return result;
    }

    /** {@code kernel(D, x)}: length-r vector with entries {@code (D_i . x + c) ^ q}. */
    static double[] kernelDx(DenseMatrix D, double[] x, double c, double q) {
        int r = D.numCols();
        double[] result = new double[r];
        for (int i = 0; i < r; i++) {
            result[i] = poly(dotColumn(D, i, x), c, q);
        }
        return result;
    }

    /** {@code exp(-sqDist / (2 * sigma2))}. */
    static double rbf(double sqDist, double sigma2) {
        return Math.exp(-sqDist / (2 * sigma2));
    }

    /** Squared Euclidean distance between column {@code col} of {@code D} and {@code x}. */
    private static double sqDistColumn(DenseMatrix D, int col, double[] x) {
        int m = D.numRows();
        double s = 0;
        for (int i = 0; i < m; i++) {
            double diff = D.values[i + m * col] - x[i];
            s += diff * diff;
        }
        return s;
    }

    /** {@code kernel(D, D)} under the RBF kernel: r x r matrix, {@code exp(-||D_i-D_j||^2/2sigma2)}. */
    static DenseMatrix kernelDDRbf(DenseMatrix D, double sigma2) {
        int m = D.numRows();
        int r = D.numCols();
        DenseMatrix result = new DenseMatrix(r, r);
        for (int i = 0; i < r; i++) {
            for (int j = i; j < r; j++) {
                double sqDist = 0;
                for (int k = 0; k < m; k++) {
                    double diff = D.values[k + m * i] - D.values[k + m * j];
                    sqDist += diff * diff;
                }
                double v = rbf(sqDist, sigma2);
                result.values[i + r * j] = v;
                result.values[j + r * i] = v;
            }
        }
        return result;
    }

    /** {@code kernel(D, x)} under the RBF kernel: length-r vector, {@code exp(-||D_i-x||^2/2sigma2)}. */
    static double[] kernelDxRbf(DenseMatrix D, double[] x, double sigma2) {
        int r = D.numCols();
        double[] result = new double[r];
        for (int i = 0; i < r; i++) {
            result[i] = rbf(sqDistColumn(D, i, x), sigma2);
        }
        return result;
    }

    /** Column sums of {@code A}: a 1 x A.numCols() row vector. */
    static DenseMatrix columnSums(DenseMatrix A) {
        DMatrixRMaj result = new DMatrixRMaj(1, A.numCols());
        CommonOps_DDRM.sumCols(toEjml(A), result);
        return toDense(result);
    }

    /** {@code D} (m x r) with each column {@code j} scaled by {@code row.get(0, j)} (1 x r). */
    static DenseMatrix scaleColumnsByRow(DenseMatrix D, DenseMatrix row) {
        int r = D.numCols();
        Preconditions.checkArgument(row.numRows() == 1 && row.numCols() == r);
        DMatrixRMaj diagMatrix = CommonOps_DDRM.diag(row.values);
        DMatrixRMaj result = new DMatrixRMaj(D.numRows(), r);
        CommonOps_DDRM.mult(toEjml(D), diagMatrix, result);
        return toDense(result);
    }

    /** {@code A * v}, where {@code A} is p x q and {@code v} has length q. */
    static double[] matVec(DenseMatrix A, double[] v) {
        Preconditions.checkArgument(v.length == A.numCols());
        DMatrixRMaj vm = new DMatrixRMaj(v.length, 1, false, v);
        DMatrixRMaj result = new DMatrixRMaj(A.numRows(), 1);
        CommonOps_DDRM.mult(toEjml(A), vm, result);
        return result.data;
    }

    /** {@code A * B}, where {@code A} is p x q and {@code B} is q x s. */
    static DenseMatrix matmul(DenseMatrix A, DenseMatrix B) {
        Preconditions.checkArgument(B.numRows() == A.numCols());
        DMatrixRMaj result = new DMatrixRMaj(A.numRows(), B.numCols());
        CommonOps_DDRM.mult(toEjml(A), toEjml(B), result);
        return toDense(result);
    }

    /** Elementwise {@code A .* B} (same shape). */
    static DenseMatrix hadamard(DenseMatrix A, DenseMatrix B) {
        Preconditions.checkArgument(A.numRows() == B.numRows() && A.numCols() == B.numCols());
        DMatrixRMaj result = toEjml(A).copy();
        CommonOps_DDRM.elementMult(result, toEjml(B));
        return toDense(result);
    }

    /** {@code sa * A + sb * B} (same shape). */
    static DenseMatrix addScaled(DenseMatrix A, double sa, DenseMatrix B, double sb) {
        Preconditions.checkArgument(A.numRows() == B.numRows() && A.numCols() == B.numCols());
        DMatrixRMaj result = new DMatrixRMaj(A.numRows(), A.numCols());
        CommonOps_DDRM.add(sa, toEjml(A), sb, toEjml(B), result);
        return toDense(result);
    }

    static DenseMatrix transpose(DenseMatrix A) {
        DMatrixRMaj result = new DMatrixRMaj(A.numCols(), A.numRows());
        CommonOps_DDRM.transpose(toEjml(A), result);
        return toDense(result);
    }

    static DenseMatrix identity(int n) {
        return toDense(CommonOps_DDRM.identity(n));
    }

    /** {@code D} (m x r) with each column {@code j} scaled by {@code diag.get(j, j)}. */
    static DenseMatrix scaleColumnsByDiag(DenseMatrix D, DenseMatrix diag) {
        int r = D.numCols();
        Preconditions.checkArgument(diag.numRows() == r && diag.numCols() == r);
        // Only the diagonal entries matter (per contract above); callers such as
        // PolyKernel#dictionaryGradient pass a full (non-diagonal) kernel matrix here, so a plain
        // D * diag matmul would wrongly mix in its off-diagonal entries.
        double[] diagValues = new double[r];
        for (int j = 0; j < r; j++) {
            diagValues[j] = diag.values[j + r * j];
        }
        DMatrixRMaj result = new DMatrixRMaj(D.numRows(), r);
        CommonOps_DDRM.mult(toEjml(D), CommonOps_DDRM.diag(diagValues), result);
        return toDense(result);
    }

    /** Inverts a square matrix. */
    static DenseMatrix invert(DenseMatrix A) {
        Preconditions.checkArgument(A.numRows() == A.numCols(), "Matrix must be square.");
        DMatrixRMaj result = new DMatrixRMaj(A.numRows(), A.numRows());
        boolean ok = CommonOps_DDRM.invert(toEjml(A), result);
        Preconditions.checkArgument(ok, "Matrix is singular.");
        return toDense(result);
    }

    /** Spectral norm (largest singular value) of a matrix, via EJML's SVD-backed induced 2-norm. */
    static double spectralNorm(DenseMatrix A) {
        return NormOps_DDRM.normP2(toEjml(A));
    }
}
