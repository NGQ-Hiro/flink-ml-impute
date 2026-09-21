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

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.RowUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements Online KFMC (poly-kernel Online Kernelized Factorization Matrix
 * Completion), a streaming version of Algorithm 2 from Fan &amp; Udell, "Online high rank matrix
 * completion" (CVPR 2019). Missing entries in each incoming {@link DenseVector} sample (marked as
 * {@link Double#NaN}) are imputed via a per-sample Newton loop against a shared dictionary {@code
 * D}, and {@code D} is updated once per mini-batch via a relaxed-Newton, momentum step. Impute and
 * train are merged: the imputed row is emitted as soon as it is computed, there is no separate
 * transform/predict pass in this (v1) implementation.
 *
 * <p>See <a href="https://arxiv.org/abs/2002.02671">Fan &amp; Udell, Online high rank matrix
 * completion, CVPR 2019</a>.
 */
public class OnlineKFMC
        implements Estimator<OnlineKFMC, OnlineKFMCModel>, OnlineKFMCParams<OnlineKFMC> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    private static final OutputTag<DenseMatrix[]> ACCUMULATOR_TAG =
            new OutputTag<DenseMatrix[]>(
                    "kfmc-accumulators",
                    ObjectArrayTypeInfo.getInfoFor(TypeInformation.of(DenseMatrix.class))) {};

    public OnlineKFMC() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public OnlineKFMCModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        int featuresIdx = inputTypeInfo.getFieldIndex(getFeaturesCol());

        // Full input row (not just featuresCol) so non-features columns (e.g. ids/timestamps
        // set by the caller) ride through the iteration body untouched and land back on the
        // imputed output row -- avoids an external zip-by-arrival-order to reattach them.
        // Identity map with an explicit RowTypeInfo: tEnv.toDataStream alone yields
        // ExternalTypeInfo (Table API's internal type), not a RowTypeInfo, and the iteration
        // body below needs a real RowTypeInfo to read field names/types off of.
        DataStream<Row> points = tEnv.toDataStream(inputs[0]).map(row -> row, inputTypeInfo);

        Table initModelDataTable =
                OnlineKFMCModelDataUtil.generateInitModelData(
                        tEnv,
                        getDims(),
                        getRank(),
                        getC(),
                        getQ(),
                        getBeta(),
                        getKernelType(),
                        getSigma2(),
                        getSeed());
        DataStream<OnlineKFMCModelData> initModelData =
                OnlineKFMCModelDataUtil.getModelDataStream(initModelDataTable);
        initModelData.getTransformation().setParallelism(1);

        IterationBody body =
                new KFMCIterationBody(
                        featuresIdx,
                        getOutputCol(),
                        getGlobalBatchSize(),
                        getC(),
                        getQ(),
                        getAlpha(),
                        getBeta(),
                        getNIter(),
                        getEta(),
                        getGamma(),
                        getMissingInit(),
                        getKernelType(),
                        getSigma2());

        DataStreamList result =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(initModelData), DataStreamList.of(points), body);
        DataStream<OnlineKFMCModelData> onlineModelData = result.get(0);
        DataStream<Row> imputedData = result.get(1);

        Table onlineModelDataTable = tEnv.fromDataStream(onlineModelData);
        String[] imputedColNames = ArrayUtils.add(inputTypeInfo.getFieldNames(), getOutputCol());
        Table imputedDataTable =
                tEnv.fromDataStream(imputedData)
                        .as(
                                imputedColNames[0],
                                ArrayUtils.remove(imputedColNames, 0));

        OnlineKFMCModel model =
                new OnlineKFMCModel().setModelData(onlineModelDataTable).setImputedData(imputedDataTable);
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    private static class KFMCIterationBody implements IterationBody {
        private final int featuresIdx;
        private final String outputCol;
        private final int batchSize;
        private final double c;
        private final double q;
        private final double alpha;
        private final double beta;
        private final int nIter;
        private final double eta;
        private final double gamma;
        private final String missingInit;
        private final String kernelType;
        private final double sigma2;

        private KFMCIterationBody(
                int featuresIdx,
                String outputCol,
                int batchSize,
                double c,
                double q,
                double alpha,
                double beta,
                int nIter,
                double eta,
                double gamma,
                String missingInit,
                String kernelType,
                double sigma2) {
            this.featuresIdx = featuresIdx;
            this.outputCol = outputCol;
            this.batchSize = batchSize;
            this.c = c;
            this.q = q;
            this.alpha = alpha;
            this.beta = beta;
            this.nIter = nIter;
            this.eta = eta;
            this.gamma = gamma;
            this.missingInit = missingInit;
            this.kernelType = kernelType;
            this.sigma2 = sigma2;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<OnlineKFMCModelData> modelData = variableStreams.get(0);
            DataStream<Row> points = dataStreams.get(0);
            int parallelism = points.getParallelism();
            Preconditions.checkState(
                    parallelism <= batchSize,
                    "There are more subtasks in the training process than the number "
                            + "of elements in each batch. Some subtasks might be idling forever.");

            RowTypeInfo pointTypeInfo = (RowTypeInfo) points.getType();
            TypeInformation<?> featuresType = pointTypeInfo.getTypeAt(featuresIdx);
            RowTypeInfo outputTypeInfo =
                    new RowTypeInfo(
                            ArrayUtils.add(pointTypeInfo.getFieldTypes(), featuresType),
                            ArrayUtils.add(pointTypeInfo.getFieldNames(), outputCol));

            SingleOutputStreamOperator<Row> localResult =
                    DataStreamUtils.generateBatchData(points, parallelism, batchSize)
                            .connect(modelData.broadcast())
                            .transform(
                                    "LocalImputeAndAccumulate",
                                    outputTypeInfo,
                                    new CalculateLocalImputeAndAccumulate(
                                            featuresIdx, c, q, nIter, eta, gamma, missingInit, kernelType, sigma2))
                            .setParallelism(parallelism);

            DataStream<Row> imputedRows = localResult;
            DataStream<DenseMatrix[]> accumulators = localResult.getSideOutput(ACCUMULATOR_TAG);

            DataStream<DenseMatrix[]> summedAccumulators =
                    accumulators
                            .countWindowAll(parallelism)
                            .reduce(
                                    (a, b) -> {
                                        addInPlace(a[0], b[0]);
                                        addInPlace(a[1], b[1]);
                                        addInPlace(a[6], b[6]);
                                        if (b[2] == null) {
                                            b[2] = a[2];
                                        }
                                        // Merge last-seen (index 5) using pre-merge counts
                                        // (index 4) before those counts get summed below;
                                        // best-effort under parallelism, see class javadoc.
                                        mergeLastSeen(a[4], a[5], b[4], b[5]);
                                        addInPlace(a[3], b[3]);
                                        addInPlace(a[4], b[4]);
                                        return b;
                                    });

            DataStream<OnlineKFMCModelData> feedbackModelData =
                    summedAccumulators
                            .transform(
                                    "UpdateDictionary",
                                    TypeInformation.of(OnlineKFMCModelData.class),
                                    new UpdateDictionary(c, q, alpha, beta, eta, gamma, kernelType, sigma2))
                            .setParallelism(1);

            return new IterationBodyResult(
                    DataStreamList.of(feedbackModelData),
                    DataStreamList.of(feedbackModelData, imputedRows));
        }

        private static void addInPlace(DenseMatrix from, DenseMatrix into) {
            for (int i = 0; i < into.values.length; i++) {
                into.values[i] += from.values[i];
            }
        }

        /**
         * Keeps {@code bLastSeen} where {@code bCount} shows this worker observed the feature
         * this round, otherwise falls back to {@code aLastSeen}. No well-defined global arrival
         * order exists across concurrent workers, so this is a best-effort choice (see {@code
         * tmp/online-kfmc-plan.md} §10 caveat), not exact last-seen-in-arrival-order semantics.
         */
        private static void mergeLastSeen(
                DenseMatrix aCount, DenseMatrix aLastSeen, DenseMatrix bCount, DenseMatrix bLastSeen) {
            for (int i = 0; i < bLastSeen.values.length; i++) {
                if (bCount.values[i] <= 0 && aCount.values[i] > 0) {
                    bLastSeen.values[i] = aLastSeen.values[i];
                }
            }
        }
    }

    /**
     * Two-input operator (par=N): for each row in a mini-batch chunk, runs the Newton-impute loop
     * against the broadcast dictionary and emits the imputed row immediately (main output); after
     * the chunk is exhausted, emits the chunk's partial gradient accumulators (side output) for
     * the central {@link UpdateDictionary} step.
     */
    private static class CalculateLocalImputeAndAccumulate extends AbstractStreamOperator<Row>
            implements TwoInputStreamOperator<Row[], OnlineKFMCModelData, Row> {
        private static final double CONVERGENCE_TOL = 1e-5;

        private final int featuresIdx;
        private final int nIter;
        private final double eta;
        private final double gamma;
        private final String missingInit;
        private final Kernel kernel;

        private transient ListState<OnlineKFMCModelData> modelDataState;
        private transient ListState<Row[]> localBatchDataState;

        private CalculateLocalImputeAndAccumulate(
                int featuresIdx,
                double c,
                double q,
                int nIter,
                double eta,
                double gamma,
                String missingInit,
                String kernelType,
                double sigma2) {
            this.featuresIdx = featuresIdx;
            this.nIter = nIter;
            this.eta = eta;
            this.gamma = gamma;
            this.missingInit = missingInit;
            this.kernel = Kernel.of(kernelType, c, q, sigma2);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            modelDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "modelData", OnlineKFMCModelData.class));
            TypeInformation<Row[]> type =
                    ObjectArrayTypeInfo.getInfoFor(TypeInformation.of(Row.class));
            localBatchDataState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("localBatch", type));
        }

        @Override
        public void processElement1(StreamRecord<Row[]> pointsRecord) throws Exception {
            localBatchDataState.add(pointsRecord.getValue());
            run();
        }

        @Override
        public void processElement2(StreamRecord<OnlineKFMCModelData> modelDataRecord)
                throws Exception {
            modelDataState.add(modelDataRecord.getValue());
            run();
        }

        private void run() throws Exception {
            if (!modelDataState.get().iterator().hasNext()
                    || !localBatchDataState.get().iterator().hasNext()) {
                return;
            }
            OnlineKFMCModelData modelData =
                    OperatorStateUtils.getUniqueElement(modelDataState, "modelData").get();
            modelDataState.clear();

            List<Row[]> chunks = IteratorUtils.toList(localBatchDataState.get().iterator());
            Row[] points = chunks.remove(0);
            localBatchDataState.update(chunks);

            DenseMatrix D = modelData.dictionary;
            DenseMatrix invKdd = modelData.kernelInverse;
            int dims = D.numRows();
            int rank = D.numCols();

            DenseMatrix accumA = new DenseMatrix(dims, rank);
            DenseMatrix accumB = new DenseMatrix(rank, rank);
            // Only populated (non-zero) when kernelType=RBF: Sigma_i T1_i (1 x rank), the RBF
            // dictionary-gradient's row accumulator; see UpdateDictionary's RBF branch.
            DenseMatrix accum2 = new DenseMatrix(1, rank);
            // Per-feature partial accumulators (dims x 1) for the missing-cell init state
            // (ModelData.featureMean / featureCount / featureLastSeen), folded centrally in
            // UpdateDictionary. Only observed (non-NaN) entries contribute.
            DenseMatrix accumSum = new DenseMatrix(dims, 1);
            DenseMatrix accumCount = new DenseMatrix(dims, 1);
            DenseMatrix accumLastSeen = new DenseMatrix(dims, 1);

            for (Row point : points) {
                DenseVector features = point.getFieldAs(featuresIdx);
                double[] x0 = features.values;
                boolean[] observed = new boolean[dims];
                double[] x = new double[dims];
                boolean anyMissing = false;
                for (int i = 0; i < dims; i++) {
                    if (Double.isNaN(x0[i])) {
                        anyMissing = true;
                        // Seeds only the working vector `x` for the Newton loop below, never
                        // `x0`/`observed` (the clamp target restoring observed entries each
                        // iteration) -- see OnlineKFMCParams#MISSING_INIT javadoc.
                        x[i] = fillMissing(i, modelData);
                    } else {
                        observed[i] = true;
                        x[i] = x0[i];
                        accumSum.values[i] += x0[i];
                        accumCount.values[i] += 1;
                        accumLastSeen.values[i] = x0[i];
                    }
                }

                double[] z;
                if (anyMissing) {
                    z = KFMCMath.matVec(invKdd, kernel.kernelDx(D, x));
                    double[] vx = new double[dims];
                    for (int iter = 0; iter < nIter; iter++) {
                        double[] Kdx = kernel.kernelDx(D, x);
                        double[] zNew = KFMCMath.matVec(invKdd, Kdx);

                        double[] gX = kernel.xGradient(D, x, Kdx, zNew, gamma);
                        for (int i = 0; i < dims; i++) {
                            vx[i] = eta * vx[i] + gX[i];
                        }
                        double[] xNew = new double[dims];
                        for (int i = 0; i < dims; i++) {
                            xNew[i] = observed[i] ? x0[i] : x[i] - vx[i];
                        }

                        double maxDiff = 0;
                        for (int i = 0; i < rank; i++) {
                            maxDiff = Math.max(maxDiff, Math.abs(z[i] - zNew[i]));
                        }
                        for (int i = 0; i < dims; i++) {
                            maxDiff = Math.max(maxDiff, Math.abs(x[i] - xNew[i]));
                        }

                        x = xNew;
                        z = zNew;
                        if (maxDiff < CONVERGENCE_TOL) {
                            break;
                        }
                    }
                } else {
                    z = KFMCMath.matVec(invKdd, kernel.kernelDx(D, x));
                }

                kernel.accumulateDictionaryGradient(D, x, z, accumA, accum2, dims, rank);
                for (int a = 0; a < rank; a++) {
                    double za = z[a];
                    for (int b = 0; b < rank; b++) {
                        accumB.values[a + rank * b] += za * z[b];
                    }
                }

                output.collect(new StreamRecord<>(RowUtils.append(point, new DenseVector(x))));
            }

            if (points.length > 0) {
                output.collect(
                        ACCUMULATOR_TAG,
                        new StreamRecord<>(
                                new DenseMatrix[] {
                                    accumA,
                                    accumB,
                                    (getRuntimeContext().getIndexOfThisSubtask() == 0) ? D : null,
                                    accumSum,
                                    accumCount,
                                    accumLastSeen,
                                    accum2
                                }));
            }
        }

        /**
         * Fills a missing (NaN) entry for the pre-Newton-loop working vector {@code x}, per
         * {@link OnlineKFMCParams#MISSING_INIT}. Falls back to 0 if the feature has no
         * observation yet, regardless of the selected mode.
         */
        private double fillMissing(int i, OnlineKFMCModelData modelData) {
            if (modelData.featureCount.values[i] <= 0) {
                return 0;
            }
            if (OnlineKFMCParams.MEAN.equals(missingInit)) {
                return modelData.featureMean.values[i];
            }
            if (OnlineKFMCParams.LAST_SEEN.equals(missingInit)) {
                return modelData.featureLastSeen.values[i];
            }
            return 0;
        }
    }

    /**
     * Central operator (par=1): given the round's summed gradient accumulators, applies the
     * relaxed-Newton momentum update to the dictionary {@code D}, recomputes {@code kernel(D, D)}
     * and its inverse, and emits the new {@link OnlineKFMCModelData}.
     */
    private static class UpdateDictionary extends AbstractStreamOperator<OnlineKFMCModelData>
            implements OneInputStreamOperator<DenseMatrix[], OnlineKFMCModelData> {
        private final double alpha;
        private final double beta;
        private final double eta;
        private final double gamma;
        private final Kernel kernel;

        private transient ListState<DenseMatrix> vDState;
        private DenseMatrix vD;
        private long modelVersion = 0L;

        private transient ListState<DenseVector> featureMeanState;
        private transient ListState<DenseVector> featureCountState;
        private transient ListState<DenseVector> featureLastSeenState;
        private DenseVector featureMean;
        private DenseVector featureCount;
        private DenseVector featureLastSeen;

        private UpdateDictionary(
                double c,
                double q,
                double alpha,
                double beta,
                double eta,
                double gamma,
                String kernelType,
                double sigma2) {
            this.alpha = alpha;
            this.beta = beta;
            this.eta = eta;
            this.gamma = gamma;
            this.kernel = Kernel.of(kernelType, c, q, sigma2);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            vDState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("vD", DenseMatrix.class));
            featureMeanState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("featureMean", DenseVector.class));
            featureCountState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("featureCount", DenseVector.class));
            featureLastSeenState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "featureLastSeen", DenseVector.class));
        }

        @Override
        public void processElement(StreamRecord<DenseMatrix[]> streamRecord) throws Exception {
            DenseMatrix[] accumulators = streamRecord.getValue();
            DenseMatrix accumA = accumulators[0];
            DenseMatrix accumB = accumulators[1];
            DenseMatrix D = accumulators[2];
            DenseMatrix partialSum = accumulators[3];
            DenseMatrix partialCount = accumulators[4];
            DenseMatrix partialLastSeen = accumulators[5];
            DenseMatrix accum2 = accumulators[6];
            int rank = D.numCols();
            int dims = D.numRows();

            if (vD == null) {
                vD = new DenseMatrix(dims, rank);
                vDState.add(vD);
            }
            if (featureMean == null) {
                featureMean = new DenseVector(dims);
                featureCount = new DenseVector(dims);
                featureLastSeen = new DenseVector(dims);
                featureMeanState.add(featureMean);
                featureCountState.add(featureCount);
                featureLastSeenState.add(featureLastSeen);
            }

            DenseMatrix identity = KFMCMath.identity(rank);
            Kernel.DictionaryGradient dictionaryGradient =
                    kernel.dictionaryGradient(D, accumA, accumB, accum2, identity, alpha, gamma, rank);
            DenseMatrix gD = dictionaryGradient.gD;
            double tau = dictionaryGradient.tau;
            if (tau < 1e-12) {
                tau = 1e-12;
            }
            for (int i = 0; i < gD.values.length; i++) {
                gD.values[i] /= tau;
            }

            for (int i = 0; i < vD.values.length; i++) {
                vD.values[i] = eta * vD.values[i] + gD.values[i];
            }
            DenseMatrix newD = KFMCMath.addScaled(D, 1.0, vD, -1.0);

            DenseMatrix newKdd = kernel.kernelDD(newD);
            DenseMatrix newC = KFMCMath.invert(KFMCMath.addScaled(newKdd, 1.0, identity, beta));

            // Fold this round's partial per-feature observations into the running missing-cell
            // init state (mean/count/last-seen), broadcast back via ModelData for the local
            // operator's fillMissing step next round.
            for (int i = 0; i < dims; i++) {
                double newCount = featureCount.values[i] + partialCount.values[i];
                if (partialCount.values[i] > 0) {
                    featureMean.values[i] =
                            (featureMean.values[i] * featureCount.values[i] + partialSum.values[i])
                                    / newCount;
                    featureLastSeen.values[i] = partialLastSeen.values[i];
                }
                featureCount.values[i] = newCount;
            }

            modelVersion++;
            output.collect(
                    new StreamRecord<>(
                            new OnlineKFMCModelData(
                                    newD,
                                    newC,
                                    featureMean.clone(),
                                    featureCount.clone(),
                                    featureLastSeen.clone(),
                                    modelVersion)));
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static OnlineKFMC load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
