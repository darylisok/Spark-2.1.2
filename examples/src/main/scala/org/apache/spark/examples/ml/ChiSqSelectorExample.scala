/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
// $example off$
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.feature.{ChiSqSelector => OldChiSqSelector}

object ChiSqSelectorExample {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local")
      .config("spark.sql.warehouse.dir", "./tmp-warehouse")
      .appName("ChiSqSelectorExample")
      .getOrCreate()
    import spark.implicits._

    // $example on$
    val data = Seq(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 18.1),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 12.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 15.0)
    )

    val df = spark.createDataset(data).toDF("id", "features", "clicked")

    val selector = new ChiSqSelector()
      .setSelectorType(OldChiSqSelector.Percentile)
      .setPercentile(0.25)
//      .setFpr(0.8)
//      .setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures")

    val result = selector.fit(df).transform(df)

    val col = selector.fit(df).labelCol

    println(col.name)
    println(selector.getFpr)
    println(selector.getNumTopFeatures)
    println(selector.getPercentile)


    df.show()
    println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
    result.show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
