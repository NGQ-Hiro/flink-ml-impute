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

package org.apache.flink.ml.examples.feature;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.feature.onlinekfmc.OnlineKFMC;
import org.apache.flink.ml.feature.onlinekfmc.OnlineKFMCModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs {@link OnlineKFMC} against the Beijing air-quality CSV (columns {@code c0}..{@code c10}),
 * imputing the dataset's real missing (empty-string) cells. Prints the original (with NaN) vs.
 * imputed vector for the first few rows that had a missing value.
 */
public class OnlineKFMCBeijingExample {
    private static final int DIMS = 11;
    private static final int MAX_ROWS = Integer.MAX_VALUE;

    public static void main(String[] args) throws Exception {
        String csvPath = args.length > 0 ? args[0] : "beijing.csv";
        List<Row> rows = loadRows(csvPath);
        System.out.println("Loaded " + rows.size() + " rows from " + csvPath);

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        RowTypeInfo typeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {DenseVectorTypeInfo.INSTANCE}, new String[] {"features"});
        DataStream<Row> trainStream = env.fromCollection(rows, typeInfo);
        Table trainTable = tEnv.fromDataStream(trainStream).as("features");

        OnlineKFMC onlineKFMC =
                new OnlineKFMC()
                        .setDims(DIMS)
                        .setRank(5)
                        .setKernelType("POLY")
                        .setC(1.0)
                        .setQ(2)
                        .setAlpha(0.1)
                        .setBeta(0.1)
                        .setNIter(20)
                        .setEta(0.5)
                        .setGamma(1.1)
                        .setMissingInit("MEAN")
                        .setGlobalBatchSize(100)
                        .setSeed(2024L)
                        .setFeaturesCol("features")
                        .setOutputCol("imputed");

        OnlineKFMCModel model = onlineKFMC.fit(trainTable);
        Table outputTable = model.transform()[0];

        String outPath = args.length > 1 ? args[1] : "beijing_imputed.csv";
        int written = 0;
        try (CloseableIterator<Row> it = outputTable.execute().collect();
                BufferedWriter writer = new BufferedWriter(new FileWriter(outPath))) {
            writer.write("c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,had_missing");
            writer.newLine();
            while (it.hasNext()) {
                Row row = it.next();
                DenseVector original = row.getFieldAs("features");
                DenseVector imputed = row.getFieldAs("imputed");
                StringBuilder sb = new StringBuilder();
                for (double d : imputed.values) {
                    sb.append(d).append(',');
                }
                sb.append(hasMissing(original));
                writer.write(sb.toString());
                writer.newLine();
                written++;
            }
        }
        System.out.println("Wrote " + written + " imputed rows to " + outPath);
    }

    private static boolean hasMissing(DenseVector v) {
        for (double d : v.values) {
            if (Double.isNaN(d)) {
                return true;
            }
        }
        return false;
    }

    /** Reads up to {@link #MAX_ROWS} rows' columns {@code c0}..{@code c10} (CSV columns 3-13). */
    private static List<Row> loadRows(String csvPath) throws Exception {
        List<Row> rows = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(csvPath))) {
            reader.readLine(); // header
            String line;
            while ((line = reader.readLine()) != null && rows.size() < MAX_ROWS) {
                String[] cols = line.split(",", -1);
                double[] values = new double[DIMS];
                for (int i = 0; i < DIMS; i++) {
                    String cell = cols[3 + i].trim();
                    values[i] =
                            (cell.isEmpty() || cell.equalsIgnoreCase("NA") || cell.equalsIgnoreCase("NaN"))
                                    ? Double.NaN
                                    : Double.parseDouble(cell);
                }
                rows.add(Row.of(new DenseVector(values)));
            }
        }
        return rows;
    }
}
