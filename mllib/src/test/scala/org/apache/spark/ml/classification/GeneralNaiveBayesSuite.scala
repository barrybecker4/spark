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

package org.apache.spark.ml.classification

import scala.util.Random

import org.apache.spark.{SparkException, SparkFunSuite}
import org.apache.spark.ml.classification.GeneralNaiveBayesSuite._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class GeneralNaiveBayesSuite extends SparkFunSuite
  with MLlibTestSparkContext with DefaultReadWriteTest {

  import testImplicits._

  @transient var dataset: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    dataset = generateSmallRandomNaiveBayesInput(42).toDF()
  }

  /**
   * @param predictionAndLabels the predictions with label
   * @param expPctCorrect the expected number of correct predictions.
   */
  def validatePrediction(predictionAndLabels: DataFrame, expPctCorrect: Double): Unit = {
    val numOfCorrectPredictions = predictionAndLabels.collect().count {
      case Row(prediction: Double, label: Double) =>
        prediction == label
    }
    // At least expPctCorrect of the predictions should be correct.
    assert(numOfCorrectPredictions > (expPctCorrect * predictionAndLabels.count()))
  }

  def validateModelFit(expLabelWeights: Array[Double],
                       expModelData: Array[Array[Array[Double]]],
                       expProbData: Array[Array[Array[Double]]],
                       model: GeneralNaiveBayesModel): Unit = {
    assert(0.1 ~== 0.1001  absTol 0.01, "mismatch") // approx equal

    assert(model.labelWeights.toArray === expLabelWeights)
    assert(model.laplaceSmoothing === 1.0)

    val expNumFeatures = expModelData.length
    val expNumClasses = expModelData(0)(0).length
    assert(model.numClasses === expNumClasses)
    assert(model.numFeatures === expNumFeatures)

    assert(model.modelData === expModelData)
    assert(model.probabilityData === expProbData)
  }

  test("params") {
    ParamsSuite.checkParams(new GeneralNaiveBayes)

    val model = new GeneralNaiveBayesModel("gnb",
      labelWeights = Vectors.dense(Array(0.2, 0.7, 0.1)),
      // Dimensions are [featureIdx][featureValue][weight for label(i)]
      modelData = Array(
        Array(Array(0.0, 1.0, 1.0), Array(1.0, 1.0, 1.0), Array(1.0, 1.0, 1.0)),
        Array(Array(1.0, 1.0, 1.0), Array(1.0, 1.0, 1.0), Array(1.0, 1.0, 1.0)),
        Array(Array(1.0, 1.0, 1.0), Array(1.0, 1.0, 1.0)),
        Array(
          Array(1.0, 1.0, 1.0), Array(1.0, 1.0, 1.0), Array(1.0, 1.0, 1.0), Array(1.0, 1.0, 1.0)
        )
      ),
      laplaceSmoothing = 0.2)

    ParamsSuite.checkParams(model)
  }

  test("naive bayes: default params") {
    val nb = new GeneralNaiveBayes
    assert(nb.getLabelCol === "label")
    assert(nb.getFeaturesCol === "features")
    assert(nb.getPredictionCol === "prediction")
    assert(nb.getSmoothing === 1.0)
  }

  test("Naive Bayes Small with random Labels") {

    val testDataset =
      generateSmallRandomNaiveBayesInput(42).toDF()
    val nb = new GeneralNaiveBayes().setSmoothing(1.0)
    val model = nb.fit(testDataset)

    val expLabelWeights = Array(2.0, 1.0, 3.0)
    val expModelData = Array(
      Array(Array(1.0, 1.0, 1.0), Array(0.0, 0.0, 1.0), Array(1.0, 0.0, 1.0)),
      Array(Array(1.0, 0.0, 0.0), Array(0.0, 1.0, 1.0), Array(1.0, 0.0, 2.0)),
      Array(Array(2.0, 1.0, 2.0), Array(0.0, 0.0, 1.0)),
      Array(Array(1.0, 0.0, 0.0), Array(0.0, 1.0, 0.0), Array(1.0, 0.0, 2.0), Array(0.0, 0.0, 1.0))
    )

    val expProbData = Array(
      Array(
        Array(0.4, 0.5, 0.3333333333333333),
        Array(0.2, 0.25, 0.3333333333333333),
        Array(0.4, 0.25, 0.3333333333333333)),
      Array(
        Array(0.4, 0.25, 0.16666666666666666),
        Array(0.2, 0.5, 0.3333333333333333),
        Array(0.4, 0.25, 0.5)),
      Array(
        Array(0.6, 0.5, 0.5),
        Array(0.2, 0.25, 0.3333333333333333)),
      Array(
        Array(0.4, 0.25, 0.16666666666666666),
        Array(0.2, 0.5, 0.16666666666666666),
        Array(0.4, 0.25, 0.5),
        Array(0.2, 0.25, 0.3333333333333333))
    )

    validateModelFit(expLabelWeights, expModelData, expProbData, model)
    assert(model.hasParent)

    val validationDataset =
      generateSmallRandomNaiveBayesInput(17).toDF()

    val predictionAndLabels: DataFrame =
      model.transform(validationDataset).select("prediction", "label")

    // Since the labels are random, we do not expect high accuracy
    validatePrediction(predictionAndLabels, 0.2)
  }

  test("Naive Bayes on Typical Data") {

    val testDataset =
      generateTypicalNaiveBayesInput().toDF()
    val nb = new GeneralNaiveBayes().setSmoothing(1.0)
    val model = nb.fit(testDataset)

    val expLabelWeights = Array(7.0, 3.0)
    val expModelData = Array(
      Array(
        Array(0.0, 0.0),
        Array(1.0, 0.0),
        Array(1.0, 0.0),
        Array(1.0, 0.0),
        Array(1.0, 0.0),
        Array(2.0, 3.0),
        Array(1.0, 0.0)
      ),
      Array(
        Array(0.0, 0.0),
        Array(0.0, 0.0),
        Array(1.0, 0.0),
        Array(0.0, 0.0),
        Array(1.0, 1.0),
        Array(4.0, 1.0),
        Array(0.0, 1.0),
        Array(1.0, 0.0)
      ),
      Array(
        Array(1.0, 0.0),
        Array(1.0, 0.0),
        Array(0.0, 3.0),
        Array(4.0, 0.0),
        Array(1.0, 0.0)
      ),
      Array(
        Array(0.0, 0.0),
        Array(1.0, 1.0),
        Array(2.0, 1.0),
        Array(3.0, 1.0),
        Array(0.0, 0.0),
        Array(1.0, 0.0)
      ),
      Array(
        Array(2.0, 1.0),
        Array(3.0, 2.0),
        Array(1.0, 0.0),
        Array(1.0, 0.0)
      )
    )
    val expProbData = Array(
      Array(
        Array(0.1111111111111111, 0.2),
        Array(0.2222222222222222, 0.2),
        Array(0.2222222222222222, 0.2),
        Array(0.2222222222222222, 0.2),
        Array(0.2222222222222222, 0.2),
        Array(0.3333333333333333, 0.8),
        Array(0.2222222222222222, 0.2)),
      Array(
        Array(0.1111111111111111, 0.2),
        Array(0.1111111111111111, 0.2),
        Array(0.2222222222222222, 0.2),
        Array(0.1111111111111111, 0.2),
        Array(0.2222222222222222, 0.4),
        Array(0.5555555555555556, 0.4),
        Array(0.1111111111111111, 0.4),
        Array(0.2222222222222222, 0.2)),
      Array(
        Array(0.2222222222222222, 0.2),
        Array(0.2222222222222222, 0.2),
        Array(0.1111111111111111, 0.8),
        Array(0.5555555555555556, 0.2),
        Array(0.2222222222222222, 0.2)),
      Array(
        Array(0.1111111111111111, 0.2),
        Array(0.2222222222222222, 0.4),
        Array(0.3333333333333333, 0.4),
        Array(0.4444444444444444, 0.4),
        Array(0.1111111111111111, 0.2),
        Array(0.2222222222222222, 0.2)),
      Array(
        Array(0.3333333333333333, 0.4),
        Array(0.4444444444444444, 0.6),
        Array(0.2222222222222222, 0.2),
        Array(0.2222222222222222, 0.2)
      )
    )

    // GeneralNaiveBayes.printModel(model.modelData)
    validateModelFit(expLabelWeights, expModelData, expProbData, model)
    assert(model.hasParent)

    val validationDataset = generateTypicalNaiveBayesInput().toDF()


    val predictionAndLabels: DataFrame =
      model.transform(validationDataset).select("prediction", "label")

    // expect about 90% accuracy
    validatePrediction(predictionAndLabels, 0.8)
  }

  test("Naive Bayes with weighted samples") {

    val testData = generateSmallRandomNaiveBayesInput(42).toDF()
    val (overSampledData, weightedData) =
      MLTestingUtils.genEquivalentOversampledAndWeightedInstances(testData,
        "label", "features", 42L)
    val nb = new GeneralNaiveBayes()
    val unweightedModel = nb.fit(weightedData)
    val overSampledModel = nb.fit(overSampledData)
    val weightedModel = nb.setWeightCol("weight").fit(weightedData)
    var numUnweightedVsOverSampledDifferences = 0

    for (
      i <- 0 until unweightedModel.numFeatures;
      j <- unweightedModel.modelData(i).indices;
      k <- 0 until unweightedModel.numClasses
    ) {
      // Oversampled and weighted models should be the same
      assert(weightedModel.modelData(i)(j)(k) ~==
        overSampledModel.modelData(i)(j)(k) relTol 0.001,
        s"${weightedModel.modelData(i)(j)(k)} did not match " +
          s"${overSampledModel.modelData(i)(j)(k)} at position $i, $j, $k"
      )

      // unweighted and overSampled should be different
      val unWtd = unweightedModel.modelData(i)(j)(k)
      val overSmp = overSampledModel.modelData(i)(j)(k)
      if (Math.abs(unWtd - overSmp) > 0.001) {
        numUnweightedVsOverSampledDifferences += 1
      }
    }
    assert(numUnweightedVsOverSampledDifferences > 10,
      "There were few differences between unweighted and overSampled. There should have been many.")
  }

  test("detect negative values") {
    val dense = spark.createDataFrame(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0)),
      LabeledPoint(0.0, Vectors.dense(-1.0)),
      LabeledPoint(1.0, Vectors.dense(1.0)),
      LabeledPoint(1.0, Vectors.dense(0.0))))
    intercept[SparkException] {
      new GeneralNaiveBayes().fit(dense)
    }
    val sparse = spark.createDataFrame(Seq(
      LabeledPoint(1.0, Vectors.sparse(1, Array(0), Array(1.0))),
      LabeledPoint(0.0, Vectors.sparse(1, Array(0), Array(-1.0))),
      LabeledPoint(1.0, Vectors.sparse(1, Array(0), Array(1.0))),
      LabeledPoint(1.0, Vectors.sparse(1, Array.empty, Array.empty))))
    intercept[SparkException] {
      new GeneralNaiveBayes().fit(sparse)
    }
    val nan = spark.createDataFrame(Seq(
      LabeledPoint(1.0, Vectors.sparse(1, Array(0), Array(1.0))),
      LabeledPoint(0.0, Vectors.sparse(1, Array(0), Array(Double.NaN))),
      LabeledPoint(1.0, Vectors.sparse(1, Array(0), Array(1.0))),
      LabeledPoint(1.0, Vectors.sparse(1, Array.empty, Array.empty))))
    intercept[SparkException] {
      new GeneralNaiveBayes().fit(nan)
    }
  }

  test("read/write") {
    def checkModelData(model: GeneralNaiveBayesModel, model2: GeneralNaiveBayesModel): Unit = {
      assert(model.labelWeights === model2.labelWeights)
      assert(model.predictionCol === model2.predictionCol)
      assert(model.modelData === model2.modelData)
      assert(model.probabilityData === model2.probabilityData)
    }
    val nb = new GeneralNaiveBayes()
    testEstimatorAndModelReadWrite(nb,
      dataset, GeneralNaiveBayesSuite.allParamSettings, checkModelData)
  }

  test("should support all NumericType labels and not support other types") {
    val nb = new GeneralNaiveBayes()
    MLTestingUtils.checkNumericTypes[GeneralNaiveBayesModel, GeneralNaiveBayes](
      nb, spark) { (expected, actual) =>
      assert(expected.labelWeights === actual.labelWeights)
      assert(expected.modelData === actual.modelData)
    }
  }
}

object GeneralNaiveBayesSuite {

  /**
   * Mapping from all Params to valid settings which differ from the defaults.
   * This is useful for tests which need to exercise all Params, such as save/load.
   * This excludes input columns to simplify some tests.
   */
  val allParamSettings: Map[String, Any] = Map(
    "predictionCol" -> "myPrediction",
    "smoothing" -> 0.1
  )

  /**
   * @param p random number in range [0, 1)
   * @param pi array of doubles [0, 1) that gives the probability distribution.
   * @return randomly selects one of the labels given the probability distribution pi
   */
  private def calcLabel(p: Double, pi: Array[Double]): Int = {
    var sum = 0.0
    for (j <- pi.indices) {
      sum += pi(j)
      if (p < sum) return j
    }
    -1
  }

  /**
   * @param seed random seed
   * @return simple data with random labels
   */
  def generateSmallRandomNaiveBayesInput(seed: Int): Seq[LabeledPoint] = {
    val numLabels = 3
    val rnd = new Random(seed)

    // This represents the raw row data. Each row has the values for each attribute
    val rawData: Array[Array[Double]] = Array(
      Array(1.0, 2.0, 0.0, 2.0),
      Array(2.0, 1.0, 1.0, 3.0),
      Array(2.0, 2.0, 0.0, 2.0),
      Array(0.0, 0.0, 0.0, 0.0),
      Array(0.0, 1.0, 0.0, 1.0),
      Array(0.0, 2.0, 0.0, 2.0)
    )

    val prob = 1.0 / numLabels
    val probs = (0 until numLabels).map(x => prob).toArray
    for (row <- rawData) yield {
      val y = calcLabel(rnd.nextDouble(), probs)
      LabeledPoint(y, Vectors.dense(row))
    }
  }

  /**
   * @return contrived data with 4 columns and 2 labels
   */
  def generateTypicalNaiveBayesInput(): Seq[LabeledPoint] = {
    val numLabels = 2
    // This represents the raw row data. Each row has the values for each attribute
    val rawData: Array[Array[Double]] = Array(
      Array(5.0, 6.0, 2.0, 3.0, 1.0),
      Array(5.0, 7.0, 3.0, 5.0, 1.0),
      Array(5.0, 5.0, 2.0, 2.0, 0.0),
      Array(5.0, 5.0, 3.0, 3.0, 2.0),
      Array(4.0, 5.0, 3.0, 3.0, 1.0),
      Array(6.0, 5.0, 3.0, 2.0, 3.0),
      Array(3.0, 5.0, 1.0, 2.0, 0.0),
      Array(1.0, 2.0, 0.0, 3.0, 0.0),
      Array(5.0, 4.0, 2.0, 1.0, 1.0),
      Array(2.0, 4.0, 4.0, 1.0, 1.0)
    )
    val labels = Array(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    for (i <- rawData.indices) yield {
      LabeledPoint(labels(i), Vectors.dense(rawData(i)))
    }
  }
}
