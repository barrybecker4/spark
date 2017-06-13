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

import scala.collection.mutable

import org.apache.hadoop.fs.Path
import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.annotation.Since
import org.apache.spark.ml.classification.GeneralNaiveBayes.MAX_EVIDENCE
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.{col, lit}


/**
 * Params for a generalized Naive Bayes Classifier.
 * Spark's original version of the Naive bayes classifier supports multinomial or bernouli versions
 * for doing document classification. This generalized version will only have one flavor and
 * be applicable to datasets with integer valued columns. If a dataset has string columns, they
 * can be converted to integer with String Indexer. If a dataset has continuous columns they
 * can be binned first so that values are replaced by an integer bin index.
 */
private[classification] trait GeneralNaiveBayesParams extends PredictorParams with HasWeightCol {

  /**
   * The smoothing parameter.
   * (default = 1.0).
   * @group param
   */
  final val smoothing: DoubleParam =
    new DoubleParam(this, "smoothing", "The (Laplace) smoothing parameter.",
    ParamValidators.gtEq(0))

  /** @group getParam */
  final def getSmoothing: Double = $(smoothing)
}

/**
 * Naive Bayes Classifiers.
 * It supports the simple bayesian classifier described
 * <a href="http://robotics.stanford.edu/~ronnyk/impSBC.pdf">here</a>.
 * The input feature values must be non-negative.
 */
@Since("2.3.0")
class GeneralNaiveBayes @Since("2.3.0") (
                                          @Since("2.3.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, GeneralNaiveBayes, GeneralNaiveBayesModel]
    with GeneralNaiveBayesParams with DefaultParamsWritable {

  import GeneralNaiveBayes._

  @Since("2.3.0")
  def this() = this(Identifiable.randomUID("gnb"))

  /**
   * Set the Laplace smoothing parameter. If non-0, prevents probabilities from becoming 0.
   * Default is 1.0.
   * @group setParam
   */
  @Since("2.3.0")
  def setSmoothing(value: Double): this.type = set(smoothing, value)
  setDefault(smoothing -> 1.0)

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  @Since("2.3.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  override protected def train(dataset: Dataset[_]): GeneralNaiveBayesModel = {
    trainWithLabelCheck(dataset)
  }

  /**
   * ml assumes input labels in range [0, numClasses).
   */
  private[spark] def trainWithLabelCheck(dataset: Dataset[_]): GeneralNaiveBayesModel = {
    val numClasses = getNumClasses(dataset)
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val numFeatures = dataset.select(col($(featuresCol))).head().getAs[Vector](0).size

    val (totalLabelWts, modelData) = getAggWeightData(dataset, numFeatures)

    val labelWts = Vectors.dense(totalLabelWts)
    new GeneralNaiveBayesModel(uid, labelWts, modelData, getSmoothing)
  }

  /**
   * @return the totalWeight for each label,
   *         and the model data that is based on weights (i.e. usually counts).
   */
  private def getAggWeightData(dataset: Dataset[_],
                               numFeatures: Int): (Array[Double], Array[Array[Array[Double]]]) = {

    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

    // Compute counts for each value of each feature, by label.
    // IOW, we need label distributions for each value of each feature.
    val zeroValue = (0.0, Array.fill(numFeatures)(mutable.Map[Double, Double]()))

    val allCounts: Array[(Double, (Double, Array[mutable.Map[Double, Double]]))] =
      dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd
        .map {
          row => (row.getDouble(0), (row.getDouble(1), row.getAs[Vector](2)))
        }
        .aggregateByKey[(Double, Array[mutable.Map[Double, Double]])](zeroValue) (
        seqOp = {
          case ((weightSum: Double, countMap: Array[mutable.Map[Double, Double]]),
                (weight, features)) =>
            requireNonnegativeValues(features) // minor performance issue?
            var index = 0
            // this way seems better, but it did not handle 0's correctly
            // features.foreachActive((index, value) => {
            features.toArray.foreach(value => {
              val valueMap = countMap(index)
              valueMap(value) = if (valueMap.contains(value)) valueMap(value) + weight else weight
              index += 1
            })
            (weightSum + weight, countMap)
        },
        combOp = {
          case ((weightSum1, countMap1), (weightSum2, countMap2)) =>
            val mergedCountMap = countMap1.indices.map(i => {
              countMap1(i) ++ countMap2(i).map {
                case (key, wt) => key -> (wt + countMap1(i).getOrElse(key, 0.0))
              }
            }).toArray
            (weightSum1 + weightSum2, mergedCountMap)
        }
      ).sortBy(_._1).collect()

    // Transform the count data to a more palatable form
    val modelData: Array[Array[Array[Double]]] = Array.ofDim[Array[Array[Double]]](numFeatures)
    val numClasses: Int = allCounts.map(_._1.toInt).max + 1

    // First find the maximum value idx for each feature. This tells us the number of values
    val featureMaxValues: Array[Array[Double]] = Array.ofDim[Double](numFeatures, numClasses)
    allCounts.foreach {
      case (label, (wt, countMap)) =>
        for (featureIdx <- countMap.indices) {
          featureMaxValues(featureIdx.toInt)(label.toInt) = countMap(featureIdx).keys.max
        }
    }
    val numFeatureValues: Array[Double] = featureMaxValues.map(_.max + 1)

    // allocate modelData array
    val firstCountMap = allCounts(0)._2._2
    firstCountMap.indices.foreach(i => {
      val cm = firstCountMap(i)
      val numValues: Int = numFeatureValues(i).toInt
      val weightData = Array.ofDim[Double](numValues, numClasses)
      modelData(i) = weightData
    })
    val totalWtForLabel: Array[Double] = Array.ofDim[Double](numClasses)
    allCounts.foreach { case (label, (wt, countMap)) => totalWtForLabel(label.toInt) = wt }

    // fill in modelData
    allCounts.foreach {
      case (labelIdx, (wt, countMap)) =>
        countMap.indices.foreach(i => {
          val cm = countMap(i)
          val weightData = modelData(i)
          val label = labelIdx.toInt
          for (valIdx <- 0 until numFeatureValues(i).toInt) {
            val wt = cm.getOrElse(valIdx, 0.0)
            if (wt > 0) {
              weightData(valIdx)(label) += wt
            }
          }
        })
    }

    (totalWtForLabel, modelData)
  }

  @Since("2.3.0")
  override def copy(extra: ParamMap): GeneralNaiveBayes = defaultCopy(extra)
}


@Since("2.3.0")
object GeneralNaiveBayes extends DefaultParamsReadable[GeneralNaiveBayes] {

  /**
   * The evidence value to use if the conditional probability is exactly 1 (rare)
   * The evidence is actually infinite in this case, but it's better to limit it
   * to allow the small possibility of other class values in the prediction.
   */
  val MAX_EVIDENCE = 1000.0

  private[GeneralNaiveBayes] def requireNonnegativeValues(v: Vector): Unit = {
    val values = v match {
      case sv: SparseVector => sv.values
      case dv: DenseVector => dv.values
    }

    require(values.forall(_ >= 0.0),
      s"General Naive Bayes requires non-negative feature values but found $v.")
  }

  @Since("2.3.0")
  override def load(path: String): GeneralNaiveBayes = super.load(path)

  /** pretty print model data for debugging purposes */
  def printModel(model: Array[Array[Array[Double]]]): Unit = {
    // scalastyle:off println println(...)
    println("modelData = \n" +
      model.map(
        "feature " + _.map(_.mkString(", ")).mkString("values: ", " / ", "")
      ).mkString("\n")
    )
    // scalastyle:on println println(...)
  }
}


/**
 * Model produced by [[GeneralNaiveBayes]]
 * @param uid id of the model
 * @param labelWeights weight for each class label
 * @param modelData label weight distribution for each value of each feature over.
 *            The dimensions of the 3d array are: [feature][featureValue][weight for label(i)]
 *             It will be used to create the probabilityData, which are the probabilities
 *            with laplace smoothing (if applied).
 * @param laplaceSmoothing laplace smoothing factor used to adjust the counts
 *                         and avoid 0 probabilities
 */
@Since("2.3.0")
class GeneralNaiveBayesModel private[ml] (
     @Since("2.3.0") override val uid: String,
     @Since("2.3.0") val labelWeights: Vector,
     @Since("2.3.0") val modelData: Array[Array[Array[Double]]],
     @Since("2.3.0") val laplaceSmoothing: Double)
  extends ProbabilisticClassificationModel[Vector, GeneralNaiveBayesModel]
    with GeneralNaiveBayesParams with MLWritable {

  @Since("2.3.0")
  override val numFeatures: Int = modelData.length

  @Since("2.3.0")
  override val numClasses: Int = labelWeights.size

  private val totalWeight: Double = labelWeights.toArray.sum
  private val priorProbabilities: Array[Double] = labelWeights.toArray.map(_ / totalWeight)
  private val priorEvidence: Array[Double] = priorProbabilities.map(x => -Math.log(1.0 - x))

  private val laplaceDenom = laplaceSmoothing * numClasses

  /**
   * Convert the weight (typically counts) to conditional probabilities and use
   * laplace correction if specified.
   */
  val probabilityData = modelData.map(dataForFeature => {
    dataForFeature.map(dataForValue => {
      var i = 0
      dataForValue.map(v => {
        val denom = labelWeights(i) + laplaceDenom
        i += 1
        (v + laplaceSmoothing) / denom
      })
    })
  })
  /**
   * Convert the conditional probabilities to evidences.
   * Evidence is -log(1 - conditionalProbability)
   * Take the log of the conditional probability so evidences will add.
   * This avoids numerical underflow when many features.
   * See https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
   * The MAX_EVIDENCE value will only be used in cases when there is no laplace smoothing
   * and there are no occurences of a class within a specific attribute value.
   */
  val evidenceData = probabilityData.map(_.map(_.map(prob => {
      val compConditionalProb = 1.0 - prob
      if (compConditionalProb == 0.0) MAX_EVIDENCE else -Math.log(compConditionalProb)
    }))
  )

  /**
   * Applies the model. The result is a conditional probability distribution for class values.
   * For each feature value, multiply conditional probabilities together to
   * get a final raw distribution
   * @return raw, unnormalized relative probabilities.
   */
  override protected def predictRaw(features: Vector): Vector = {
    val evidence: Array[Double] = priorEvidence.clone()
    var featureIdx = 0
    // features.foreachActive((featureIdx, value) => {  // this should work but gave wrong results
    features.toArray.foreach(value => {
      val v = value.toInt
      val featureEvidence = evidenceData(featureIdx)
      // There may occasionally be values in the test data that were not in the training data.
      // In these cases v will be >= featureProbs.length. Such values are ignored.
      if (v < featureEvidence.length) {
        val md = featureEvidence(v)
        featureIdx += 1
        for (i <- 0 until numClasses) {
          evidence(i) += md(i)
        }
      }
    })
    Vectors.dense(evidence)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in GeneralNaiveBayesModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  @Since("2.3.0")
  override def copy(extra: ParamMap): GeneralNaiveBayesModel = {
    copyValues(new GeneralNaiveBayesModel(uid, labelWeights, modelData, laplaceSmoothing)
      .setParent(this.parent), extra)
  }

  @Since("2.3.0")
  override def toString: String = {
    s"GeneralNaiveBayesModel (uid=$uid) with $numClasses classes"
  }

  @Since("2.3.0")
  override def write: MLWriter = new GeneralNaiveBayesModel.GeneralNaiveBayesModelWriter(this)
}


@Since("2.3.0")
object GeneralNaiveBayesModel extends MLReadable[GeneralNaiveBayesModel] {

  @Since("2.3.0")
  override def read: MLReader[GeneralNaiveBayesModel] = new GeneralNaiveBayesModelReader

  @Since("2.3.0")
  override def load(path: String): GeneralNaiveBayesModel = super.load(path)

  /** [[MLWriter]] instance for [[GeneralNaiveBayesModel]] */
  private[GeneralNaiveBayesModel] class GeneralNaiveBayesModelWriter(
             instance: GeneralNaiveBayesModel) extends MLWriter {

    private case class Data(labelWeights: Vector, modelDataStr: String, laplaceSmoothing: Double)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)

      // Write the multi-dimensional model as json
      val modelDataStr = convertToJson(instance.modelData)
      val data = Data(instance.labelWeights, modelDataStr, instance.laplaceSmoothing)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private def convertToJson(mData: Array[Array[Array[Double]]]): String = {
    mData.map(
      _.map(_.mkString("[", ",", "]")).mkString("[", ",", "]")
    )
      .mkString("[", ",", "]")
  }


  private class GeneralNaiveBayesModelReader extends MLReader[GeneralNaiveBayesModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GeneralNaiveBayesModel].getName

    override def load(path: String): GeneralNaiveBayesModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val Row(labelWeights: Vector, modelDataStr: String, laplaceSmoothing: Double) =
        MLUtils.convertVectorColumnsToML(data, "labelWeights")
          .select("labelWeights", "modelDataStr", "laplaceSmoothing")
          .head()

      // extract the model from the json that represents the multi-dimensional array as a string
      implicit val formats = DefaultFormats
      val modelData: Array[Array[Array[Double]]] =
        parse(modelDataStr).extract[Array[Array[Array[Double]]]]
      // println(" READ modelDataJson ")

      val model = new GeneralNaiveBayesModel(metadata.uid,
        labelWeights, modelData, laplaceSmoothing)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
