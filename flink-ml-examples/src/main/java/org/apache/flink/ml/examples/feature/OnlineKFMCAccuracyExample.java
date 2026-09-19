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
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Accuracy check: real beijing.csv has no ground truth for its missing cells, so this takes
 * fully-observed rows, artificially masks one column per row (recorded as truth), runs {@link
 * OnlineKFMC}, then matches each imputed row back to its source row by its unmasked entries
 * (same technique as {@code OnlineKFMCTest#relativeMaskedError}) and reports relative L2 error
 * between imputed and true values, overall and per column.
 */
public class OnlineKFMCAccuracyExample {
    private static final int DIMS = 11;
    private static final int MAX_ROWS = 3000;

    public static void main(String[] args) throws Exception {
        String csvPath = args.length > 0 ? args[0] : "beijing.csv";
        List<double[]> trueRows = loadFullyObservedRows(csvPath);
        System.out.println("Loaded " + trueRows.size() + " fully-observed rows from " + csvPath);

        Random rnd = new Random(2024L);
        int[] maskedIdx = new int[trueRows.size()];
        List<Row> masked = new ArrayList<>();
        for (int i = 0; i < trueRows.size(); i++) {
            double[] row = trueRows.get(i).clone();
            int m = rnd.nextInt(DIMS);
            maskedIdx[i] = m;
            row[m] = Double.NaN;
            masked.add(Row.of(new DenseVector(row)));
        }

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        RowTypeInfo typeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {DenseVectorTypeInfo.INSTANCE}, new String[] {"features"});
        DataStream<Row> trainStream = env.fromCollection(masked, typeInfo);
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
                        .setMissingInit("LAST_SEEN")
                        .setGlobalBatchSize(100)
                        .setSeed(2024L)
                        .setFeaturesCol("features")
                        .setOutputCol("imputed");

        OnlineKFMCModel model = onlineKFMC.fit(trainTable);
        Table outputTable = model.transform()[0];

        double[] sqErrByCol = new double[DIMS];
        double[] sqTrueByCol = new double[DIMS];
        long[] countByCol = new long[DIMS];
        int matched = 0;
        try (CloseableIterator<Row> it = outputTable.execute().collect()) {
            while (it.hasNext()) {
                Row row = it.next();
                DenseVector imputed = row.getFieldAs("imputed");
                int best = nearestRow(imputed.values, trueRows, maskedIdx);
                if (best < 0) {
                    continue;
                }
                int m = maskedIdx[best];
                double truth = trueRows.get(best)[m];
                double diff = imputed.values[m] - truth;
                sqErrByCol[m] += diff * diff;
                sqTrueByCol[m] += truth * truth;
                countByCol[m]++;
                matched++;
            }
        }

        System.out.println("Matched " + matched + " / " + trueRows.size() + " rows");
        double totalSqErr = 0;
        double totalSqTrue = 0;
        for (int c = 0; c < DIMS; c++) {
            totalSqErr += sqErrByCol[c];
            totalSqTrue += sqTrueByCol[c];
            if (countByCol[c] > 0) {
                double relErr = Math.sqrt(sqErrByCol[c] / sqTrueByCol[c]);
                System.out.printf(
                        "c%d: n=%d relative_L2_error=%.4f%n", c, countByCol[c], relErr);
            }
        }
        System.out.printf(
                "OVERALL: relative_L2_error=%.4f%n", Math.sqrt(totalSqErr / totalSqTrue));
    }

    /** Finds the row in trueRows whose unmasked entries best match imputed's unmasked entries. */
    private static int nearestRow(double[] imputed, List<double[]> trueRows, int[] maskedIdx) {
        int best = -1;
        double bestDist = Double.MAX_VALUE;
        for (int i = 0; i < trueRows.size(); i++) {
            double[] truth = trueRows.get(i);
            int m = maskedIdx[i];
            double dist = 0;
            for (int j = 0; j < DIMS; j++) {
                if (j != m) {
                    double d = imputed[j] - truth[j];
                    dist += d * d;
                }
            }
            if (dist < bestDist) {
                bestDist = dist;
                best = i;
            }
        }
        return best;
    }

    /** Reads up to {@link #MAX_ROWS} rows that have no missing value in any of c0..c10. */
    private static List<double[]> loadFullyObservedRows(String csvPath) throws Exception {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(csvPath))) {
            reader.readLine(); // header
            String line;
            while ((line = reader.readLine()) != null && rows.size() < MAX_ROWS) {
                String[] cols = line.split(",", -1);
                if (cols.length < 3 + DIMS) {
                    continue;
                }
                double[] values = new double[DIMS];
                boolean complete = true;
                for (int i = 0; i < DIMS; i++) {
                    String cell = cols[3 + i].trim();
                    if (cell.isEmpty() || cell.equalsIgnoreCase("NA") || cell.equalsIgnoreCase("NaN")) {
                        complete = false;
                        break;
                    }
                    values[i] = Double.parseDouble(cell);
                }
                if (complete) {
                    rows.add(values);
                }
            }
        }
        return rows;
    }
}
