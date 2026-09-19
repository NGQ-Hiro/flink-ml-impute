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

import org.apache.flink.ml.common.param.HasGlobalBatchSize;
import org.apache.flink.ml.common.param.HasSeed;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params of {@link OnlineKFMC}.
 *
 * @param <T> The class type of this instance.
 */
public interface OnlineKFMCParams<T>
        extends HasGlobalBatchSize<T>, HasSeed<T>, OnlineKFMCModelParams<T> {

    String ZERO = "ZERO";
    String MEAN = "MEAN";
    String LAST_SEEN = "LAST_SEEN";

    String POLY = "POLY";
    String RBF = "RBF";

    /**
     * Supported options of the missing-cell init strategy, applied to a sample's {@link
     * Double#NaN} entries before the Newton-impute loop starts.
     *
     * <ul>
     *   <li>ZERO: fill with 0 (matches the reference implementation).
     *   <li>MEAN: fill with the feature's running mean of observed values seen so far.
     *   <li>LAST_SEEN: fill with the feature's last observed value (forward-fill).
     * </ul>
     *
     * <p>Regardless of the selected mode, a feature with no observation yet falls back to 0.
     */
    Param<String> MISSING_INIT =
            new StringParam(
                    "missingInit",
                    "Strategy to fill missing entries with before the Newton-impute loop.",
                    LAST_SEEN,
                    ParamValidators.inArray(ZERO, MEAN, LAST_SEEN));

    default String getMissingInit() {
        return get(MISSING_INIT);
    }

    default T setMissingInit(String value) {
        return set(MISSING_INIT, value);
    }

    /** Row dimension {@code m} of each sample / the dictionary {@code D}. */
    Param<Integer> DIMS =
            new IntParam("dims", "Row dimension of each sample and of the dictionary.", 1, ParamValidators.gt(0));

    /** Dictionary width / kernel feature rank {@code r}. */
    Param<Integer> RANK =
            new IntParam("rank", "Rank of the dictionary D.", 1, ParamValidators.gt(0));

    /** Poly kernel additive constant {@code c} in {@code (x^T y + c) ^ q}. */
    Param<Double> C = new DoubleParam("c", "Poly kernel additive constant.", 1.0);

    /** Poly kernel exponent {@code q} in {@code (x^T y + c) ^ q}. */
    Param<Integer> Q = new IntParam("q", "Poly kernel exponent.", 2, ParamValidators.gt(0));

    /** Kernel type: {@code POLY} (default) or {@code RBF}. */
    Param<String> KERNEL_TYPE =
            new StringParam(
                    "kernelType", "Kernel type: POLY or RBF.", POLY, ParamValidators.inArray(POLY, RBF));

    /**
     * RBF kernel bandwidth {@code sigma2} in {@code exp(-||x-y||^2 / (2 * sigma2))}. Only used
     * when {@code kernelType=RBF}. Fixed, user-supplied value (v2 does not implement the
     * reference's auto-bandwidth estimation from data variance).
     */
    Param<Double> SIGMA2 =
            new DoubleParam(
                    "sigma2", "RBF kernel bandwidth (only used when kernelType=RBF).", 1.0, ParamValidators.gt(0.0));

    /** Regularization on the dictionary {@code D}. */
    Param<Double> ALPHA =
            new DoubleParam("alpha", "Regularization parameter of the dictionary D.", 0.1, ParamValidators.gtEq(0.0));

    /** Regularization on the code {@code Z} (added to {@code kernel(D, D)} before inverting). */
    Param<Double> BETA =
            new DoubleParam("beta", "Regularization parameter of the code Z.", 0.1, ParamValidators.gtEq(0.0));

    /** Inner Newton-impute iterations per sample. */
    Param<Integer> N_ITER =
            new IntParam("nIter", "Number of inner Newton-impute iterations per sample.", 50, ParamValidators.gt(0));

    /** Momentum coefficient. */
    Param<Double> ETA =
            new DoubleParam("eta", "Momentum coefficient.", 0.5, ParamValidators.gtEq(0.0));

    /** Relaxed-Newton step-size scale. */
    Param<Double> GAMMA =
            new DoubleParam("gamma", "Relaxed-Newton step-size scale.", 1.1, ParamValidators.gt(0.0));

    default int getDims() {
        return get(DIMS);
    }

    default T setDims(Integer value) {
        return set(DIMS, value);
    }

    default int getRank() {
        return get(RANK);
    }

    default T setRank(Integer value) {
        return set(RANK, value);
    }

    default double getC() {
        return get(C);
    }

    default T setC(Double value) {
        return set(C, value);
    }

    default int getQ() {
        return get(Q);
    }

    default T setQ(Integer value) {
        return set(Q, value);
    }

    default String getKernelType() {
        return get(KERNEL_TYPE);
    }

    default T setKernelType(String value) {
        return set(KERNEL_TYPE, value);
    }

    default double getSigma2() {
        return get(SIGMA2);
    }

    default T setSigma2(Double value) {
        return set(SIGMA2, value);
    }

    default double getAlpha() {
        return get(ALPHA);
    }

    default T setAlpha(Double value) {
        return set(ALPHA, value);
    }

    default double getBeta() {
        return get(BETA);
    }

    default T setBeta(Double value) {
        return set(BETA, value);
    }

    default int getNIter() {
        return get(N_ITER);
    }

    default T setNIter(Integer value) {
        return set(N_ITER, value);
    }

    default double getEta() {
        return get(ETA);
    }

    default T setEta(Double value) {
        return set(ETA, value);
    }

    default double getGamma() {
        return get(GAMMA);
    }

    default T setGamma(Double value) {
        return set(GAMMA, value);
    }
}
