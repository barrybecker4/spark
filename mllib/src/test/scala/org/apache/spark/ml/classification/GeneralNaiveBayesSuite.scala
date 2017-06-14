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
    val totalRows = predictionAndLabels.count()
    val numExpectedCorrectPredictions = expPctCorrect * totalRows
    assert(numOfCorrectPredictions > numExpectedCorrectPredictions,
      s"Expected at least $numExpectedCorrectPredictions out of $totalRows " +
        s"to be correct, but got only $numOfCorrectPredictions")
  }

  def validateModelFit(expLabelWeights: Array[Double],
                       expModelData: Array[Array[Array[Double]]],
                       expLogProbabilityData: Option[Array[Array[Array[Double]]]],
                       model: GeneralNaiveBayesModel,
                       expLaplaceSmoothing: Double = 1.0): Unit = {
    assert(0.1 ~== 0.1001  absTol 0.01, "mismatch") // approx equal

    assert(model.labelWeights.toArray === expLabelWeights)
    assert(model.laplaceSmoothing === expLaplaceSmoothing)

    val expNumFeatures = expModelData.length
    val expNumClasses = expModelData(0)(0).length
    assert(model.numClasses === expNumClasses)
    assert(model.numFeatures === expNumFeatures)

    assert(model.modelData === expModelData)
    if (expLogProbabilityData.isDefined) {
      assert(model.logProbabilityData === expLogProbabilityData.get)
    }
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

    val expLogProbabilityData = Array(
      Array(
        Array(-0.916290731874155, -0.6931471805599453, -1.0986122886681098),
        Array(-1.6094379124341003, -1.3862943611198906, -1.0986122886681098),
        Array(-0.916290731874155, -1.3862943611198906, -1.0986122886681098)),
      Array(
        Array(-0.916290731874155, -1.3862943611198906, -1.791759469228055),
        Array(-1.6094379124341003, -0.6931471805599453, -1.0986122886681098),
        Array(-0.916290731874155, -1.3862943611198906, -0.6931471805599453)),
      Array(
        Array(-0.5108256237659907, -0.6931471805599453, -0.6931471805599453),
        Array(-1.6094379124341003, -1.3862943611198906, -1.0986122886681098)),
      Array(
        Array(-0.916290731874155, -1.3862943611198906, -1.791759469228055),
        Array(-1.6094379124341003, -0.6931471805599453, -1.791759469228055),
        Array(-0.916290731874155, -1.3862943611198906, -0.6931471805599453),
        Array(-1.6094379124341003, -1.3862943611198906, -1.0986122886681098))
    )

    validateModelFit(expLabelWeights, expModelData, Some(expLogProbabilityData), model)
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
    val expLogProbabilityData = Array(
      Array(
        Array(-2.1972245773362196, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003),
        Array(-1.0986122886681098, -0.2231435513142097),
        Array(-1.5040773967762742, -1.6094379124341003)),
      Array(
        Array(-2.1972245773362196, -1.6094379124341003),
        Array(-2.1972245773362196, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003),
        Array(-2.1972245773362196, -1.6094379124341003),
        Array(-1.5040773967762742, -0.916290731874155),
        Array(-0.587786664902119, -0.916290731874155),
        Array(-2.1972245773362196, -0.916290731874155),
        Array(-1.5040773967762742, -1.6094379124341003)),
      Array(
        Array(-1.5040773967762742, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003),
        Array(-2.1972245773362196, -0.2231435513142097),
        Array(-0.587786664902119, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003)),
      Array(
        Array(-2.1972245773362196, -1.6094379124341003),
        Array(-1.5040773967762742, -0.916290731874155),
        Array(-1.0986122886681098, -0.916290731874155),
        Array(-0.8109302162163288, -0.916290731874155),
        Array(-2.1972245773362196, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003)),
      Array(Array(-1.0986122886681098, -0.916290731874155),
        Array(-0.8109302162163288, -0.5108256237659907),
        Array(-1.5040773967762742, -1.6094379124341003),
        Array(-1.5040773967762742, -1.6094379124341003))
    )



    // GeneralNaiveBayes.printModel(model.modelData)
    validateModelFit(expLabelWeights, expModelData, Some(expLogProbabilityData), model)
    assert(model.hasParent)

    val validationDataset = generateTypicalNaiveBayesInput().toDF()


    val predictionAndLabels: DataFrame =
      model.transform(validationDataset).select("prediction", "label")

    validatePrediction(predictionAndLabels, 0.8)
  }

  test("Naive Bayes on potentially underflowing (numRows = 20 numCols = 10)") {

    val numRows = 20
    val numColumns = 10
    val testDataset =
      generatePotentialUnderflowNaiveBayesInput(numRows, numColumns).toDF()
    val laplaceSmoothing = 0.01
    val nb = new GeneralNaiveBayes().setSmoothing(laplaceSmoothing)
    val model = nb.fit(testDataset)

    val expLabelWeights = Array(17.0, 3.0)
    val expModelData = Array(
      Array(Array(15.0, 1.0), Array(2.0, 2.0)),
      Array(Array(17.0, 0.0), Array(0.0, 3.0)),
      Array(Array(17.0, 1.0), Array(0.0, 2.0)),
      Array(Array(15.0, 0.0), Array(2.0, 3.0)),
      Array(Array(15.0, 0.0), Array(2.0, 3.0)),
      Array(Array(15.0, 0.0), Array(2.0, 3.0)),
      Array(Array(17.0, 0.0), Array(0.0, 3.0)),
      Array(Array(15.0, 0.0), Array(2.0, 3.0)),
      Array(Array(17.0, 0.0), Array(0.0, 3.0)),
      Array(Array(12.0, 0.0), Array(5.0, 3.0))
    )

    val expEvidenveData = Array(
      Array(
        Array(-0.1256724774998575, -1.0953065005336102),
        Array(-2.1362544010742437, -0.40712210931579396)),
      Array(
        Array(-5.87716737457604E-4, -5.71042701737487),
        Array(-7.43955930913332, -0.003316752625994038)),
      Array(
        Array(-5.87716737457604E-4, -1.0953065005336102),
        Array(-7.43955930913332, -0.40712210931579396)),
      Array(
        Array(-0.1256724774998575, -5.71042701737487),
        Array(-2.1362544010742437, -0.003316752625994038)),
      Array(
        Array(-0.1256724774998575, -5.71042701737487),
        Array(-2.1362544010742437, -0.003316752625994038)),
      Array(
        Array(-0.1256724774998575, -5.71042701737487),
        Array(-2.1362544010742437, -0.003316752625994038)),
      Array(
        Array(-5.87716737457604E-4, -5.71042701737487),
        Array(-7.43955930913332, -0.003316752625994038)),
      Array(
        Array(-0.1256724774998575, -5.71042701737487),
        Array(-2.1362544010742437, -0.003316752625994038)),
      Array(
        Array(-5.87716737457604E-4, -5.71042701737487),
        Array(-7.43955930913332, -0.003316752625994038)),
      Array(
        Array(-0.3486494870533359, -5.71042701737487),
        Array(-1.2229532080484546, -0.003316752625994038)
      )
    )


    validateModelFit(expLabelWeights, expModelData, Some(expEvidenveData), model, laplaceSmoothing)
    assert(model.hasParent)

    val validationDataset =
      generatePotentialUnderflowNaiveBayesInput(numRows, numColumns).toDF()
    val predictionAndLabels: DataFrame =
      model.transform(validationDataset).select("prediction", "label")

    validatePrediction(predictionAndLabels, 0.99)
  }


  test("Naive Bayes on potentially underflowing (numRows = 10 numCols = 5 no lapalce)") {

    val numRows = 10
    val numColumns = 5
    val testDataset =
      generatePotentialUnderflowNaiveBayesInput(numRows, numColumns).toDF()
    val laplaceSmoothing = 0.0
    val nb = new GeneralNaiveBayes().setSmoothing(laplaceSmoothing)
    val model = nb.fit(testDataset)

    val expLabelWeights = Array(6.0, 4.0)
    val expModelData = Array(
      Array(Array(6.0, 1.0), Array(0.0, 3.0)),
      Array(Array(5.0, 0.0), Array(1.0, 4.0)),
      Array(Array(6.0, 0.0), Array(0.0, 4.0)),
      Array(Array(5.0, 0.0), Array(1.0, 4.0)),
      Array(Array(5.0, 0.0), Array(1.0, 4.0))
    )

    val expLogProbabilityData = Array(
      Array(Array(0.0, -1.3862943611198906), Array(-1000.0, -0.2876820724517809)),
      Array(Array(-0.1823215567939546, -1000.0), Array(-1.791759469228055, 0.0)),
      Array(Array(0.0, -1000.0), Array(-1000.0, 0.0)),
      Array(Array(-0.1823215567939546, -1000.0), Array(-1.791759469228055, 0.0)),
      Array(Array(-0.1823215567939546, -1000.0), Array(-1.791759469228055, 0.0))
    )


    validateModelFit(expLabelWeights, expModelData, Some(expLogProbabilityData), model, laplaceSmoothing)
    assert(model.hasParent)

    val validationDataset =
      generatePotentialUnderflowNaiveBayesInput(numRows, numColumns).toDF()
    val predictionAndLabels: DataFrame =
      model.transform(validationDataset).select("prediction", "label")

    validatePrediction(predictionAndLabels, 0.99)
  }

  test("Naive Bayes on potentially underflowing (numRows = 2000 numCols = 20)") {

    val numRows = 2000
    val numColumns = 20
    val testDataset =
      generatePotentialUnderflowNaiveBayesInput(numRows, numColumns).toDF()
    val laplaceSmoothing = 0.01
    val nb = new GeneralNaiveBayes().setSmoothing(laplaceSmoothing)
    val model = nb.fit(testDataset)

    val expLabelWeights = Array(1582.0, 418.0)
    val expModelData = Array(
      Array(Array(1396.0, 42.0), Array(186.0, 376.0)),
      Array(Array(1425.0, 44.0), Array(157.0, 374.0)),
      Array(Array(1425.0, 50.0), Array(157.0, 368.0)),
      Array(Array(1417.0, 41.0), Array(165.0, 377.0)),
      Array(Array(1391.0, 47.0), Array(191.0, 371.0)),
      Array(Array(1431.0, 47.0), Array(151.0, 371.0)),
      Array(Array(1406.0, 38.0), Array(176.0, 380.0)),
      Array(Array(1405.0, 40.0), Array(177.0, 378.0)),
      Array(Array(1419.0, 36.0), Array(163.0, 382.0)),
      Array(Array(1431.0, 44.0), Array(151.0, 374.0)),
      Array(Array(1411.0, 47.0), Array(171.0, 371.0)),
      Array(Array(1429.0, 49.0), Array(153.0, 369.0)),
      Array(Array(1403.0, 36.0), Array(179.0, 382.0)),
      Array(Array(1422.0, 47.0), Array(160.0, 371.0)),
      Array(Array(1426.0, 39.0), Array(156.0, 379.0)),
      Array(Array(1411.0, 42.0), Array(171.0, 376.0)),
      Array(Array(1416.0, 43.0), Array(166.0, 375.0)),
      Array(Array(1419.0, 43.0), Array(163.0, 375.0)),
      Array(Array(1428.0, 43.0), Array(154.0, 375.0)),
      Array(Array(1436.0, 45.0), Array(146.0, 373.0))
    )

    validateModelFit(expLabelWeights, expModelData, None, model, laplaceSmoothing)
    assert(model.hasParent)

    val validationDataset =
      generatePotentialUnderflowNaiveBayesInput(numRows, numColumns).toDF()

    val predictionAndLabels: DataFrame =
      model.transform(validationDataset).select("prediction", "label")

    validatePrediction(predictionAndLabels, 0.99)
  }

  test("Naive Bayes on potentially underflowing " +
    "(numRows = 500 numCols = 2000 probDeviation = 40%)") {

    val numRows = 500
    val numColumns = 2000
    val proportionLabel1 = 0.9
    val probDeviation = 0.40
    val testDataset =
      generatePotentialUnderflowNaiveBayesInput(numRows, numColumns,
        proportionLabel1, probDeviation).toDF()
    val laplaceSmoothing = 0.01
    val nb = new GeneralNaiveBayes().setSmoothing(laplaceSmoothing)
    val model = nb.fit(testDataset)

    val validationDataset =
      generatePotentialUnderflowNaiveBayesInput(numRows, numColumns,
        proportionLabel1, probDeviation).toDF()

    val predictionAndLabels: DataFrame =
      model.transform(validationDataset).select("prediction", "label")

    // Should be at least 99% correct.  Before the change to use log probabilities,
    // the percent correct was lower because of numerical underflow.
    validatePrediction(predictionAndLabels, 0.99)
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
      assert(model.logProbabilityData === model2.logProbabilityData)
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

  /**
   * If we just multiply conditional probabilities, numbers can underflow.
   * To avoid this the code will instead add log values.
   * To simulate a case where underflow can happen, consider the following case.
   * Let's say we have a credit card fraud training data. The label is "fraud".
   * Each of the 30 contrived columns has 2 values. Each of these columns represent
   * some fictional property that if 0, then strongly tends toward not fraud, and if
   * 1, then strongly tends toward fraud. Now, if we try to make a prediction for
   * [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
   * it might predict not fraud, when it should really predict fraud because the all the small
   * probabilities initially multiplied together might underflow and become 0.
   *
   * @param numRows number of rows of fake data
   * @param numColumns of columns of fake data
   * @param proportionLabel1 proportion of rows you want to have label1 the rest will have label2
   * @param probDeviation This it the chance that the attribute values deviates
    *                      from what is expected, given the label
   * @return contrived data with 30 columns (of 2 values each) and 2 labels that
   *         will result in underflow if multiplication of probabilities instead
   *         of adding log values is used.
   */
  def generatePotentialUnderflowNaiveBayesInput(
        numRows: Int = 500, numColumns: Int = 40,
        proportionLabel1: Double = 0.8,
        probDeviation: Double = 0.1): Seq[LabeledPoint] = {
    val numLabels = 2
    val rng = new Random(seed = 1)

    val dat = for (i <- 0 until numRows) yield {
      val rndNum = rng.nextDouble()
      val label = if (rndNum < proportionLabel1) 0.0 else 1.0
      val func: (Int) => Double = {
        if (label == 0) (x) => if (rng.nextDouble() < probDeviation) 1.0 else 0.0
        else (x) => if (rng.nextDouble() < probDeviation) 0.0 else 1.0
      }

      val rawData = Array.tabulate[Double](numColumns)(func)
      // println("raw = " + rawData.mkString(", "))
      LabeledPoint(label, Vectors.dense(rawData))
    }
    // scalastyle:off println println()
    // println("raw dat = " + dat.map(x => x.toString).mkString("\n "))
    dat
  }
}
