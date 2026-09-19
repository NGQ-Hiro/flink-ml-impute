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

package org.apache.flink.ml.feature;

import org.apache.flink.api.common.JobID;
import org.apache.flink.api.common.JobSubmissionResult;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.ml.feature.onlinekfmc.OnlineKFMC;
import org.apache.flink.ml.feature.onlinekfmc.OnlineKFMCModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.util.InMemorySinkFunction;
import org.apache.flink.ml.util.InMemorySourceFunction;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.TestLogger;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Tests {@link OnlineKFMC} and {@link OnlineKFMCModel} against a synthetic rank-1 matrix with
 * masked (NaN) entries, checking that the Newton-impute + dictionary-update loop drives the
 * imputed values towards the true, unmasked matrix over a few rounds.
 */
public class OnlineKFMCTest extends TestLogger {
    private static final int DIMS = 4;
    private static final int N_ROWS = 8;
    private static final double[] BASE = {1.0, 2.0, -1.0, 0.5};

    // Rank-1 matrix: row i = scale[i] * BASE, with one entry per row masked as NaN.
    private static final double[] SCALE = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};

    private static final Configuration config =
            new Configuration() {
                {
                    set(RestOptions.BIND_PORT, "18081-19091");
                }
            };

    private static MiniCluster miniCluster;
    private static StreamExecutionEnvironment env;
    private static StreamTableEnvironment tEnv;

    private InMemorySourceFunction<Row> trainSource;
    private InMemorySinkFunction<Row> outputSink;

    @BeforeClass
    public static void beforeClass() throws Exception {
        miniCluster =
                new MiniCluster(
                        new MiniClusterConfiguration.Builder()
                                .setConfiguration(config)
                                .setNumTaskManagers(1)
                                .setNumSlotsPerTaskManager(1)
                                .build());
        miniCluster.start();
    }

    @AfterClass
    public static void afterClass() throws Exception {
        miniCluster.close();
    }

    @Before
    public void before() {
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(1);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);

        trainSource = new InMemorySourceFunction<>();
        outputSink = new InMemorySinkFunction<>();
    }

    @After
    public void after() throws Exception {
        for (org.apache.flink.runtime.client.JobStatusMessage message :
                miniCluster.listJobs().get()) {
            miniCluster.cancelJob(message.getJobId());
        }
    }

    /** True (unmasked) row i of the synthetic rank-1 matrix. */
    private static double[] trueRow(int i) {
        double[] row = new double[DIMS];
        for (int j = 0; j < DIMS; j++) {
            row[j] = SCALE[i] * BASE[j];
        }
        return row;
    }

    /** Row i with its (i % DIMS)-th entry masked as NaN. */
    private static Row maskedRow(int i) {
        double[] row = trueRow(i);
        row[i % DIMS] = Double.NaN;
        return Row.of(new DenseVector(row));
    }

    /**
     * Matches each imputed row back to its source row by nearest observed (non-NaN) entries
     * (robust to any reordering introduced by the batch/iteration pipeline), then returns the
     * relative L2 error between the imputed and true values at the masked positions only.
     */
    private static double relativeMaskedError(List<Row> imputedRows) {
        double sqErr = 0;
        double sqTrue = 0;
        for (Row row : imputedRows) {
            double[] imputed = ((DenseVector) row.getFieldAs(1)).values;
            int bestIdx = -1;
            double bestDist = Double.MAX_VALUE;
            for (int i = 0; i < N_ROWS; i++) {
                double[] truth = trueRow(i);
                int maskIdx = i % DIMS;
                double dist = 0;
                for (int j = 0; j < DIMS; j++) {
                    if (j != maskIdx) {
                        dist += (imputed[j] - truth[j]) * (imputed[j] - truth[j]);
                    }
                }
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
            double[] truth = trueRow(bestIdx);
            int maskIdx = bestIdx % DIMS;
            double diff = imputed[maskIdx] - truth[maskIdx];
            sqErr += diff * diff;
            sqTrue += truth[maskIdx] * truth[maskIdx];
        }
        return Math.sqrt(sqErr / sqTrue);
    }

    @Test
    public void testImputeErrorDecreases() throws Exception {
        runImputeErrorDecreasesTest(new OnlineKFMC().setMissingInit("ZERO"), 1);
    }

    /**
     * Same setup as {@link #testImputeErrorDecreases}, but with {@code missingInit=MEAN}: checks
     * that the mean-seeded fill still converges. Round 1 uses an all-zero {@code featureMean}
     * (cold start, {@code featureCount==0} everywhere before the first dictionary update), so it
     * behaves identically to {@code ZERO} for that round; the mean only becomes non-trivial from
     * round 2 onward.
     */
    @Test
    public void testImputeErrorDecreasesWithMeanInit() throws Exception {
        runImputeErrorDecreasesTest(new OnlineKFMC().setMissingInit("MEAN"), 1);
    }

    /**
     * Same setup as {@link #testImputeErrorDecreases}, but with {@code kernelType=RBF}. Uses
     * rank 3 (vs. poly's rank 1): a single RBF dictionary atom can't span the range of row scales
     * in this synthetic matrix (RBF has no poly-style exponent to generalize across scale), so a
     * few atoms are needed for the kernel to interpolate.
     */
    @Test
    public void testImputeErrorDecreasesWithRbfKernel() throws Exception {
        runImputeErrorDecreasesTest(
                new OnlineKFMC().setMissingInit("ZERO").setKernelType("RBF").setSigma2(20.0), 3);
    }

    private void runImputeErrorDecreasesTest(OnlineKFMC onlineKFMCPartial, int rank) throws Exception {
        Table trainTable =
                tEnv.fromDataStream(
                        env.addSource(
                                trainSource,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            TypeInformation.of(DenseVector.class)
                                        },
                                        new String[] {"features"})));

        OnlineKFMC onlineKFMC =
                onlineKFMCPartial
                        .setDims(DIMS)
                        .setRank(rank)
                        .setC(1.0)
                        .setQ(2)
                        .setAlpha(0.1)
                        .setBeta(0.1)
                        .setNIter(50)
                        .setEta(0.5)
                        .setGamma(1.1)
                        .setGlobalBatchSize(N_ROWS)
                        .setSeed(2024L)
                        .setFeaturesCol("features")
                        .setOutputCol("output");

        OnlineKFMCModel model = onlineKFMC.fit(trainTable);
        Table outputTable = model.transform()[0];
        tEnv.toDataStream(outputTable).addSink(outputSink);

        JobGraph jobGraph = env.getStreamGraph().getJobGraph();
        JobID jobID =
                miniCluster
                        .submitJob(jobGraph)
                        .thenApply(JobSubmissionResult::getJobID)
                        .get(1, TimeUnit.SECONDS);
        Assert.assertNotNull(jobID);

        Row[] rows = new Row[N_ROWS];
        for (int i = 0; i < N_ROWS; i++) {
            rows[i] = maskedRow(i);
        }

        double firstRoundError = -1;
        double lastRoundError = -1;
        int rounds = 8;
        for (int round = 0; round < rounds; round++) {
            trainSource.addAll(rows);
            List<Row> imputed = outputSink.poll(N_ROWS);
            double error = relativeMaskedError(imputed);
            if (round == 0) {
                firstRoundError = error;
            }
            lastRoundError = error;
        }

        Assert.assertTrue(
                "Imputed relative error should decrease after training: first="
                        + firstRoundError
                        + " last="
                        + lastRoundError,
                lastRoundError < firstRoundError);
        Assert.assertTrue(
                "Imputed relative error should fall below threshold: last=" + lastRoundError,
                lastRoundError < 0.4);
    }
}
