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

import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * A Model produced by {@link OnlineKFMC}. Since v1 merges impute and train (the local operator
 * emits each imputed row while training), this Model does not run a separate prediction pass:
 * {@link #transform} simply returns the imputed-data table that was already computed during
 * {@link OnlineKFMC#fit}, regardless of the {@code inputs} argument. {@link #getModelData()}
 * additionally exposes the {@link OnlineKFMCModelData} (dictionary {@code D} and kernel inverse
 * {@code C}) stream, e.g. for checkpointed save/reload.
 */
public class OnlineKFMCModel implements Model<OnlineKFMCModel>, OnlineKFMCModelParams<OnlineKFMCModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;
    private Table imputedDataTable;

    public OnlineKFMCModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkState(
                imputedDataTable != null,
                "No imputed data available. This Model must be produced by OnlineKFMC#fit.");
        return new Table[] {imputedDataTable};
    }

    /** Sets the imputed-data table produced during {@link OnlineKFMC#fit}. */
    OnlineKFMCModel setImputedData(Table imputedDataTable) {
        this.imputedDataTable = imputedDataTable;
        return this;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                OnlineKFMCModelDataUtil.getModelDataStream(modelDataTable),
                path,
                new OnlineKFMCModelDataUtil.ModelDataEncoder());
    }

    public static OnlineKFMCModel load(StreamTableEnvironment tEnv, String path) throws IOException {
        OnlineKFMCModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new OnlineKFMCModelDataUtil.ModelDataDecoder());
        model.setModelData(modelDataTable);
        return model;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public OnlineKFMCModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }
}
