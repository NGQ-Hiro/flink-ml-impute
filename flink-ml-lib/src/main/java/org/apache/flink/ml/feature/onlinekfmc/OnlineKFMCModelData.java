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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseMatrixSerializer;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Model data of {@link OnlineKFMC} and {@link OnlineKFMCModel}: the dictionary {@code D} (m x r)
 * and the precomputed kernel inverse {@code C = (kernel(D, D) + beta * I) ^ -1} (r x r), so that
 * workers never need to redundantly invert the same r x r matrix. Also carries the running
 * per-feature statistics (each size {@code m}) used to fill missing entries before the
 * Newton-impute loop: {@code featureMean} (running mean of observed values), {@code featureCount}
 * (observation count, the mean's denominator), and {@code featureLastSeen} (last observed value,
 * forward-fill).
 */
public class OnlineKFMCModelData {

    public DenseMatrix dictionary;

    public DenseMatrix kernelInverse;

    public DenseVector featureMean;

    public DenseVector featureCount;

    public DenseVector featureLastSeen;

    public long modelVersion;

    public OnlineKFMCModelData() {}

    public OnlineKFMCModelData(
            DenseMatrix dictionary,
            DenseMatrix kernelInverse,
            DenseVector featureMean,
            DenseVector featureCount,
            DenseVector featureLastSeen,
            long modelVersion) {
        this.dictionary = dictionary;
        this.kernelInverse = kernelInverse;
        this.featureMean = featureMean;
        this.featureCount = featureCount;
        this.featureLastSeen = featureLastSeen;
        this.modelVersion = modelVersion;
    }

    @VisibleForTesting
    public void encode(OutputStream outputStream) throws IOException {
        DataOutputViewStreamWrapper out = new DataOutputViewStreamWrapper(outputStream);
        DenseMatrixSerializer matrixSerializer = new DenseMatrixSerializer();
        DenseVectorSerializer vectorSerializer = new DenseVectorSerializer();
        matrixSerializer.serialize(dictionary, out);
        matrixSerializer.serialize(kernelInverse, out);
        vectorSerializer.serialize(featureMean, out);
        vectorSerializer.serialize(featureCount, out);
        vectorSerializer.serialize(featureLastSeen, out);
        out.writeLong(modelVersion);
    }

    static OnlineKFMCModelData decode(InputStream inputStream) throws IOException {
        DataInputViewStreamWrapper in = new DataInputViewStreamWrapper(inputStream);
        DenseMatrixSerializer matrixSerializer = new DenseMatrixSerializer();
        DenseVectorSerializer vectorSerializer = new DenseVectorSerializer();
        DenseMatrix dictionary = matrixSerializer.deserialize(in);
        DenseMatrix kernelInverse = matrixSerializer.deserialize(in);
        DenseVector featureMean = vectorSerializer.deserialize(in);
        DenseVector featureCount = vectorSerializer.deserialize(in);
        DenseVector featureLastSeen = vectorSerializer.deserialize(in);
        long modelVersion = in.readLong();
        return new OnlineKFMCModelData(
                dictionary, kernelInverse, featureMean, featureCount, featureLastSeen, modelVersion);
    }
}
