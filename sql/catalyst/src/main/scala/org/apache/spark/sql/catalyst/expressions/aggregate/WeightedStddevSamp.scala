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

package org.apache.spark.sql.catalyst.expressions.aggregate

import org.apache.spark.sql.catalyst.dsl.expressions._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.{Expression, ExpressionDescription, If, Literal, Sqrt}
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.DoubleType


/**
 * Weighted standard deviation calculation.
 *
 * References:
 *  - Xiangrui Meng.  "Simpler Online Updates for Arbitrary-Order Central Moments."
 *      2015. http://arxiv.org/abs/1510.04923
 *
 * @see <a href="https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance">
 * Algorithms for calculating variance (Wikipedia)</a>
 * See http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
 * See also http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
 * See also
 * https://blog.cordiner.net/2010/06/16/calculating-variance-and-mean-with-mapreduce-python/
 *   (This gave the clue I needed for the mergeExpressions function)
 *
 * @param child column to compute central moments of.
 * @param weight the weighting column
 */
abstract class WeightedCentralMomentAgg(child: Expression, weight: Expression)
  extends DeclarativeAggregate {

  override def children: Seq[Expression] = Seq(child, weight)
  override def nullable: Boolean = true
  override def dataType: DataType = DoubleType
  override def inputTypes: Seq[AbstractDataType] = Seq(DoubleType, DoubleType)

  protected val n = AttributeReference("n", DoubleType, nullable = false)()
  protected val wSum = AttributeReference("wSum", DoubleType, nullable = false)()
  protected val mean = AttributeReference("mean", DoubleType, nullable = false)()
  protected val s = AttributeReference("s", DoubleType, nullable = false)()
  override val aggBufferAttributes = Seq(n, wSum, mean, s)
  override val initialValues: Seq[Expression] = Array.fill(4)(Literal(0.0))

  // See https://en.wikipedia.org/wiki/Algorithms_for_
  // calculating_variance#Weighted_incremental_algorithm
  override val updateExpressions: Seq[Expression] = {

    val newN = n + Literal(1.0)
    val newWSum = wSum + weight
    val newMean = mean + (weight / newWSum) * (child - mean)
    val newS = s + weight * (child - mean) * (child - newMean)

    Seq(
      If(IsNull(child), n, newN),
      If(IsNull(child), wSum, newWSum),
      If(IsNull(child), mean, newMean),
      If(IsNull(child), s, newS)
    )
  }

  override val mergeExpressions: Seq[Expression] = {   // mean.right = x    mean.left = mean
    val newN = n.left + n.right
    val wSum1 = wSum.left
    val wSum2 = wSum.right
    val newWSum = wSum1 + wSum2
    val delta = mean.right - mean.left

    val deltaN = If(newWSum === Literal(0.0), Literal(0.0), delta / newWSum)
    // val newMean = (wSum1 * mean.left + wSum2 * mean.right) /
    //   newWSum  // simpler but more expensive form
    val newMean = mean.left + deltaN * wSum2

    // Derived from
    // https://blog.cordiner.net/2010/06/16/calculating-variance-and-mean-with-mapreduce-python/
    // It gives the same result as the commented method below, but is easier to understand.
    val newS = (((wSum1 * s.left) + (wSum2 * s.right)) / newWSum) +
      (wSum1 * wSum2 * deltaN * deltaN)
    Seq(newN, newWSum, newMean, newS)
  }

//  This gives the same result as above and may even be faster, but its more confusing
//  override val mergeExpressions: Seq[Expression] = {   // mean.right = x    mean.left = mean
//    val wSum1 = wSum.left
//    val wSum2 = wSum.right
//    val newWSum = wSum1 + wSum2
//    val delta = mean.right - mean.left
//    val deltaN = If(newWSum === Literal(0.0), Literal(0.0), delta / newWSum)
//    val newMean = mean.left + deltaN * wSum2
//
//    val newS = s.left + s.right + wSum1 * wSum2  * delta * deltaN
//    //val newS = s.left + s.right + wSum1 * wSum2 * delta * deltaN
//    Seq(newWSum, newMean, newS)
//  }
}

// Compute the weighted sample standard deviation of a column
// scalastyle:off line.size.limit
@ExpressionDescription(
  usage = "_FUNC_(expr) - Returns the sample weighted standard deviation calculated from values of a group.")
// scalastyle:on line.size.limit
case class WeightedStddevSamp(child: Expression, weight: Expression)
  extends WeightedCentralMomentAgg(child, weight) {

  override val evaluateExpression: Expression = {
    If(wSum === Literal(0.0), Literal.create(null, DoubleType),
      If(wSum === Literal(1.0), Literal(Double.NaN),
        Sqrt(s / ((n - 1.0) * wSum / n)) ) )
  }

  override def prettyName: String = "wtd_stddev_samp"
}

// Compute the weighted population standard deviation of a column
// scalastyle:off line.size.limit
@ExpressionDescription(
  usage = "_FUNC_(expr) - Returns the population weighted standard deviation calculated from values of a group.")
// scalastyle:on line.size.limit
case class WeightedStddevPop(child: Expression, weight: Expression)
  extends WeightedCentralMomentAgg(child, weight) {

  override val evaluateExpression: Expression = {
    If(wSum === Literal(0.0), Literal.create(null, DoubleType),
      If(wSum === Literal(1.0), Literal(Double.NaN),
        Sqrt(s / wSum) ) )
  }

  override def prettyName: String = "wtd_stddev_samp"
}
