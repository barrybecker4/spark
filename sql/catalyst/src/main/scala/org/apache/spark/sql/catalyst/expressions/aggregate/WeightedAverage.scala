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

import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.dsl.expressions._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.util.TypeUtils
import org.apache.spark.sql.types._

@ExpressionDescription(
  usage = "_FUNC_(expr, wt) - Returns the weighted mean calculated from values of a group.")
case class WeightedAverage(child: Expression, weight: Expression) extends DeclarativeAggregate {

  override def prettyName: String = "weighted_avg"

  override def children: Seq[Expression] = child :: weight :: Nil

  override def nullable: Boolean = true

  // Return data type.
  override def dataType: DataType = resultType

  override def inputTypes: Seq[AbstractDataType] = Seq(NumericType, NumericType)

  override def checkInputDataTypes(): TypeCheckResult = {
    TypeUtils.checkForNumericExpr(child.dataType, "function weighted average")
    TypeUtils.checkForNumericExpr(weight.dataType, "function weighted average")
  }

  private lazy val resultType = child.dataType match {
    case DecimalType.Fixed(p, s) =>
      DecimalType.bounded(p + 4, s + 4)
    case _ => DoubleType
  }

  private lazy val sumDataType = child.dataType match {
    case _ @ DecimalType.Fixed(p, s) => DecimalType.bounded(p + 10, s)
    case _ => DoubleType
  }

  private lazy val weightedSum = AttributeReference("weightedSum", sumDataType)()
  private lazy val sumWeight = AttributeReference("sumWeight", DoubleType)()

  override lazy val aggBufferAttributes: List[AttributeReference] = weightedSum :: sumWeight :: Nil

  override lazy val initialValues = Seq(
    /* weightedSum = */ Cast(Literal(0.0), sumDataType),
    /* sumWeight = */ Literal(0.0)
  )

  override lazy val updateExpressions = Seq(
    /* weightedSum = */
    Add(weightedSum, If(IsNull(child), Literal(0.0), Multiply(child, weight))),
    /* sumWeight= */
    If(IsNull(child), sumWeight, Add(sumWeight, weight))
  )

  override lazy val mergeExpressions = Seq(
    /* weightedSum = */ weightedSum.left + weightedSum.right,
    /* sumWeight = */ sumWeight.left + sumWeight.right
  )

  // If all input are nulls, count will be 0 and we will get null after the division.
  override lazy val evaluateExpression = child.dataType match {
    case DecimalType.Fixed(p, s) =>
      // increase the precision and scale to prevent precision loss
      val dt = DecimalType.bounded(p + 14, s + 4)
      Cast(Cast(weightedSum, dt) / Cast(sumWeight, dt), resultType)
    case _ =>
      Cast(weightedSum, resultType) / Cast(sumWeight, resultType)
  }
}
