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

import java.io.Serializable;

/**
 * Strategy for the kernel-specific math in Online KFMC, so that adding a new kernel touches one
 * new implementation instead of every branch site in {@link OnlineKFMC}.
 */
interface Kernel extends Serializable {

    /** Dispatches {@code kernel(D, x)} to the poly or RBF kernel. */
    double[] kernelDx(DenseMatrix D, double[] x);

    /** Dispatches {@code kernel(D, D)} to the poly or RBF kernel. */
    DenseMatrix kernelDD(DenseMatrix D);

    /**
     * Per-point Newton-impute x-gradient plus its Hessian-approx {@code tau}, computed inside the
     * per-point Newton loop in {@link OnlineKFMC.CalculateLocalImputeAndAccumulate}.
     *
     * @param D dictionary
     * @param x current working vector for this point
     * @param Kdx {@code kernel(D, x)} for the current x
     * @param zNew {@code invKdd * Kdx}
     */
    double[] xGradient(DenseMatrix D, double[] x, double[] Kdx, double[] zNew, double gamma);

    /**
     * Accumulates the dictionary-gradient contribution of one converged point (final {@code x}/
     * {@code z}) into {@code accumA} (m x r) and {@code accum2} (1 x r, RBF-only; left all-zero
     * for poly).
     */
    void accumulateDictionaryGradient(
            DenseMatrix D,
            double[] x,
            double[] z,
            DenseMatrix accumA,
            DenseMatrix accum2,
            int dims,
            int rank);

    /**
     * Dictionary gradient {@code gD} plus its Hessian-approx {@code tau}, computed once per
     * mini-batch in {@link OnlineKFMC.UpdateDictionary}.
     */
    DictionaryGradient dictionaryGradient(
            DenseMatrix D,
            DenseMatrix accumA,
            DenseMatrix accumB,
            DenseMatrix accum2,
            DenseMatrix identity,
            double alpha,
            double gamma,
            int rank);

    /** Bundled result of {@link #dictionaryGradient}. */
    final class DictionaryGradient {
        final DenseMatrix gD;
        final double tau;

        DictionaryGradient(DenseMatrix gD, double tau) {
            this.gD = gD;
            this.tau = tau;
        }
    }

    /** Builds the {@link Kernel} matching {@code kernelType}, defaulting to poly. */
    static Kernel of(String kernelType, double c, double q, double sigma2) {
        return OnlineKFMCParams.RBF.equals(kernelType)
                ? new RbfKernel(sigma2)
                : new PolyKernel(c, q);
    }

    static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    static double[] hadamard(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
}
