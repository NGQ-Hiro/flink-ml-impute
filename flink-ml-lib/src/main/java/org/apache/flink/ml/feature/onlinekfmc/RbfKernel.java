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

/** RBF-kernel implementation of {@link Kernel}. */
final class RbfKernel implements Kernel {
    private final double sigma2;

    RbfKernel(double sigma2) {
        this.sigma2 = sigma2;
    }

    @Override
    public double[] kernelDx(DenseMatrix D, double[] x) {
        return KFMCMath.kernelDxRbf(D, x, sigma2);
    }

    @Override
    public DenseMatrix kernelDD(DenseMatrix D) {
        return KFMCMath.kernelDDRbf(D, sigma2);
    }

    @Override
    public double[] xGradient(DenseMatrix D, double[] x, double[] Kdx, double[] zNew, double gamma) {
        // RBF Newton-impute x-gradient (KFMC_minibatch.m lines 103-109):
        // g_Kxd = -z_new', T1 = g_Kxd .* Kdx, g_X1 = (D*T1 - x*sum(T1))/sigma2,
        // tau = gamma/sigma2 * |sum(T1)|, g_X = g_X1/tau.
        int dims = x.length;
        int rank = zNew.length;
        double[] T1 = new double[rank];
        double sumT1 = 0;
        for (int j = 0; j < rank; j++) {
            T1[j] = -zNew[j] * Kdx[j];
            sumT1 += T1[j];
        }
        double[] DT1 = KFMCMath.matVec(D, T1);
        double tau = gamma / sigma2 * Math.abs(sumT1);
        double[] gX = new double[dims];
        for (int i = 0; i < dims; i++) {
            gX[i] = (DT1[i] - x[i] * sumT1) / sigma2 / tau;
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
        // Dictionary-gradient accumulators (KFMC_minibatch.m lines 142-150, RBF
        // branch): T1 = -z .* Kdx (final x/z); accumA += x (x) T1 (Accum1, m x r),
        // accum2 += T1 (Accum2, 1 x r). accumB (Accum3 = Sigma z z^T) is shared with
        // the poly branch.
        double[] Kdx = kernelDx(D, x);
        double[] T1 = new double[rank];
        for (int b = 0; b < rank; b++) {
            T1[b] = -z[b] * Kdx[b];
            accum2.values[b] += T1[b];
        }
        for (int a = 0; a < dims; a++) {
            double xa = x[a];
            for (int b = 0; b < rank; b++) {
                accumA.values[a + dims * b] += xa * T1[b];
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
        // RBF dictionary gradient (KFMC_minibatch.m lines 142-150): g_Kdd = 0.5*accumB
        // (= Sigma z z^T) + 0.5*alpha*I, T2 = g_Kdd .* Kdd, g_D1 = (accumA - D.*accum2)/
        // sigma2 (accumA = Accum1), g_D2 = 2*(D*T2 - D.*colSums(T2))/sigma2,
        // tau = gamma*normest((2*T2 - diag(accum2 + 2*|colSums(T2)|))/sigma2).
        DenseMatrix Kdd = KFMCMath.kernelDDRbf(D, sigma2);
        DenseMatrix gKdd = KFMCMath.addScaled(accumB, 0.5, identity, 0.5 * alpha);
        DenseMatrix T2 = KFMCMath.hadamard(gKdd, Kdd);
        DenseMatrix sumT2 = KFMCMath.columnSums(T2);

        DenseMatrix gD1 =
                KFMCMath.addScaled(
                        accumA, 1.0 / sigma2, KFMCMath.scaleColumnsByRow(D, accum2), -1.0 / sigma2);
        DenseMatrix gD2 =
                KFMCMath.addScaled(
                        KFMCMath.matmul(D, T2),
                        2.0 / sigma2,
                        KFMCMath.scaleColumnsByRow(D, sumT2),
                        -2.0 / sigma2);
        DenseMatrix gD = KFMCMath.addScaled(gD1, 1.0, gD2, 1.0);

        DenseMatrix tauMat = new DenseMatrix(rank, rank);
        for (int i = 0; i < rank; i++) {
            for (int j = 0; j < rank; j++) {
                tauMat.values[i + rank * j] = 2.0 / sigma2 * T2.values[i + rank * j];
            }
            tauMat.values[i + rank * i] -=
                    (accum2.values[i] + 2.0 * Math.abs(sumT2.values[i])) / sigma2;
        }
        double tau = gamma * KFMCMath.spectralNorm(tauMat);
        return new DictionaryGradient(gD, tau);
    }
}
