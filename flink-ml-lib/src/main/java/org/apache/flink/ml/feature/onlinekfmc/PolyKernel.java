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

/** Polynomial-kernel implementation of {@link Kernel}. */
final class PolyKernel implements Kernel {
    private final double c;
    private final double q;

    PolyKernel(double c, double q) {
        this.c = c;
        this.q = q;
    }

    @Override
    public double[] kernelDx(DenseMatrix D, double[] x) {
        return KFMCMath.kernelDx(D, x, c, q);
    }

    @Override
    public DenseMatrix kernelDD(DenseMatrix D) {
        return KFMCMath.kernelDD(D, c, q);
    }

    @Override
    public double[] xGradient(DenseMatrix D, double[] x, double[] Kdx, double[] zNew, double gamma) {
        int dims = x.length;
        double dotXX = Kernel.dot(x, x);
        double w1 = KFMCMath.poly(dotXX, c, q - 1);
        double[] w2 = KFMCMath.kernelDx(D, x, c, q - 1);
        double[] dW2Z = KFMCMath.matVec(D, Kernel.hadamard(w2, zNew));

        double tau = gamma * (q * w1);
        double[] gX = new double[dims];
        for (int i = 0; i < dims; i++) {
            gX[i] = (q * x[i] * w1 - q * dW2Z[i]) / tau;
        }
        return gX;
    }

    @Override
    public void accumulateDictionaryGradient(
            DenseMatrix D,
            double[] x,
            double[] z,
            DenseMatrix accumA,
            DenseMatrix accum2,
            int dims,
            int rank) {
        double[] wD = KFMCMath.kernelDx(D, x, c, q - 1);
        double[] wz = Kernel.hadamard(wD, z);
        for (int a = 0; a < dims; a++) {
            double xa = x[a];
            for (int b = 0; b < rank; b++) {
                accumA.values[a + dims * b] += xa * wz[b];
            }
        }
    }

    @Override
    public DictionaryGradient dictionaryGradient(
            DenseMatrix D,
            DenseMatrix accumA,
            DenseMatrix accumB,
            DenseMatrix accum2,
            DenseMatrix identity,
            double alpha,
            double gamma,
            int rank) {
        DenseMatrix w2 = KFMCMath.kernelDD(D, c, q - 1);
        DenseMatrix accumBHadW2 = KFMCMath.hadamard(accumB, w2);
        DenseMatrix term2 = KFMCMath.matmul(D, accumBHadW2);
        DenseMatrix term3 = KFMCMath.scaleColumnsByDiag(D, w2);

        DenseMatrix gD = new DenseMatrix(D.numRows(), rank);
        for (int i = 0; i < gD.values.length; i++) {
            gD.values[i] =
                    -q * accumA.values[i] + q * term2.values[i] + alpha * q * term3.values[i];
        }

        DenseMatrix w2Diag = KFMCMath.hadamard(w2, identity);
        DenseMatrix hessianApprox = KFMCMath.addScaled(accumBHadW2, q, w2Diag, alpha * q);
        double tau = gamma * KFMCMath.spectralNorm(hessianApprox);
        return new DictionaryGradient(gD, tau);
    }
}
