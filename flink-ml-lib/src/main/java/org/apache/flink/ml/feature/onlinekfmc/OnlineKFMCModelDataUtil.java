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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.util.Random;

/** Utility methods to generate the initial dictionary and convert {@link OnlineKFMCModelData}. */
class OnlineKFMCModelDataUtil {

    private OnlineKFMCModelDataUtil() {}

    /**
     * Generates a Table containing one {@link OnlineKFMCModelData} whose dictionary {@code D} is
     * {@code dims x rank}, i.i.d. standard Gaussian (matches {@code KFMC_minibatch.m}'s {@code
     * randn(m, r)} init), and whose kernel inverse is the corresponding {@code (kernel(D, D) +
     * beta * I) ^ -1}. Built eagerly (par=1), no input data required.
     */
    static Table generateInitModelData(
            StreamTableEnvironment tEnv,
            int dims,
            int rank,
            double c,
            double q,
            double beta,
            String kernelType,
            double sigma2,
            long seed) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        return tEnv.fromDataStream(
                env.fromElements(1)
                        .map(new InitDictionaryGenerator(dims, rank, c, q, beta, kernelType, sigma2, seed))
                        .setParallelism(1));
    }

    private static class InitDictionaryGenerator
            implements MapFunction<Integer, OnlineKFMCModelData> {
        private final int dims;
        private final int rank;
        private final double c;
        private final double q;
        private final double beta;
        private final String kernelType;
        private final double sigma2;
        private final long seed;

        private InitDictionaryGenerator(
                int dims,
                int rank,
                double c,
                double q,
                double beta,
                String kernelType,
                double sigma2,
                long seed) {
            this.dims = dims;
            this.rank = rank;
            this.c = c;
            this.q = q;
            this.beta = beta;
            this.kernelType = kernelType;
            this.sigma2 = sigma2;
            this.seed = seed;
        }

        @Override
        public OnlineKFMCModelData map(Integer ignored) {
            Random random = new Random(seed);
            DenseMatrix D = new DenseMatrix(dims, rank);
            for (int i = 0; i < D.values.length; i++) {
                D.values[i] = random.nextGaussian();
            }
            DenseMatrix Kdd =
                    OnlineKFMCParams.RBF.equals(kernelType)
                            ? KFMCMath.kernelDDRbf(D, sigma2)
                            : KFMCMath.kernelDD(D, c, q);
            DenseMatrix C =
                    KFMCMath.invert(
                            KFMCMath.addScaled(Kdd, 1.0, KFMCMath.identity(rank), beta));
            return new OnlineKFMCModelData(
                    D, C, new DenseVector(dims), new DenseVector(dims), new DenseVector(dims), 0L);
        }
    }

    static DataStream<OnlineKFMCModelData> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable)
                .map(
                        x ->
                                new OnlineKFMCModelData(
                                        x.getFieldAs(0),
                                        x.getFieldAs(1),
                                        x.getFieldAs(2),
                                        x.getFieldAs(3),
                                        x.getFieldAs(4),
                                        x.getFieldAs(5)));
    }

    /** Data encoder for {@link OnlineKFMC} and {@link OnlineKFMCModel}. */
    static class ModelDataEncoder implements Encoder<OnlineKFMCModelData> {
        @Override
        public void encode(OnlineKFMCModelData modelData, java.io.OutputStream outputStream)
                throws IOException {
            modelData.encode(outputStream);
        }
    }

    /** Data decoder for {@link OnlineKFMC} and {@link OnlineKFMCModel}. */
    static class ModelDataDecoder extends SimpleStreamFormat<OnlineKFMCModelData> {
        @Override
        public Reader<OnlineKFMCModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<OnlineKFMCModelData>() {
                @Override
                public OnlineKFMCModelData read() throws IOException {
                    try {
                        return OnlineKFMCModelData.decode(inputStream);
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    inputStream.close();
                }
            };
        }

        @Override
        public TypeInformation<OnlineKFMCModelData> getProducedType() {
            return TypeInformation.of(OnlineKFMCModelData.class);
        }
    }
}
