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

package org.apache.spark.sql

import org.apache.spark.sql.catalyst.expressions.{Cast, Expression, If, IsNotNull, Literal}
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.plans.logical.{LocalRelation, LogicalPlan}
import org.apache.spark.sql.catalyst.util.usePrettyExpression
import org.apache.spark.sql.types.{NumericType, StringType, StructField, StructType}

/**
 * Compute basic stats for a dataframe containing only numeric columns.
 * This class is based off of the describe method in DataFrame.
 * The implementation makes use of some things private to DataFrame
 * so that's why its in this package.
 *
 * @param df data frame with all continuous columns
 * @param weightingColumn an optional weighting column.
 *        If unspecified, this is the same as df.describe
 */
class BasicStatistics(df: DataFrame, weightingColumn: Option[String]) {

  /**
   * Computes weighted statistics for numeric columns if there is a weighting column.
   * If no weighting column specified, we simply delegate to df.describe
   * to get the unweighted stats. The columns are the same as what is returned
   * by dataFrame.describe, but reflect weighted values.
   * They are:
   * count (i.e. weight),
   * mean (i.e. weightedMean),
   * stddev (i.e. weighted stddev),
   * min, and max.
   * If no columns are given, this function computes statistics for all numerical.
   */
  def describe(): DataFrame = {
    if (weightingColumn.isDefined) {
      weightedDescribe(weightingColumn.get)
    } else {
      df.describe()
    }
  }

  /** Version of describe which requires a weighting column */
  private def weightedDescribe(weightColumn: String): DataFrame = withPlan {

    // The list of summary statistics to compute, in the form of expressions.
    // Don't weight the weighting column.
    val statistics = List[(String, (Expression, Expression) => Expression)](
      "count" -> ((child: Expression, weight: Expression) => {
        if (child == weight) {
          Count(child).toAggregateExpression()
        } else {
          Sum(If(IsNotNull(child), weight, Literal(0.0))).toAggregateExpression()
        }
      }),
      "mean" -> ((child: Expression, weight: Expression) => {
        if (child == weight) {
          Average(child).toAggregateExpression()
        } else {
          WeightedAverage(child, weight).toAggregateExpression()
        }
      }),
      "stddev" -> ((child: Expression, weight: Expression) => {
        if (child == weight) {
          StddevSamp(child).toAggregateExpression()
        }
        else {
          WeightedStddevSamp(child, weight).toAggregateExpression()
        }
      }),
      "min" -> ((child: Expression, weight: Expression) => Min(child).toAggregateExpression()),
      "max" -> ((child: Expression, weight: Expression) => Max(child).toAggregateExpression())
    )

    val outputCols = aggregatableColumns.map(usePrettyExpression(_).sql).toList

    val ret: Seq[Row] = if (outputCols.nonEmpty) {
      val weightExpr = Column(weightColumn).expr
      val aggExprs = statistics.flatMap { case (_, colToAgg) =>
        outputCols.map(col =>
          Column(Cast(colToAgg(Column(col).expr, weightExpr), StringType)).as(col)
        )
      }

      val row = df.groupBy().agg(aggExprs.head, aggExprs.tail: _*).head().toSeq

      // Pivot the data so each summary is one row
      row.grouped(outputCols.size).toSeq.zip(statistics).map { case (aggregation, (statistic, _)) =>
        Row(statistic :: aggregation.toList: _*)
      }
    } else {
      // If there are no output columns, just output a single column that contains the stats.
      statistics.map { case (name, _) => Row(name) }
    }

    // All columns are string type
    val schema = StructType(
      StructField("summary", StringType) :: outputCols.map(StructField(_, StringType))).toAttributes

    // `toArray` forces materialization to make the seq serializable
    LocalRelation.fromExternalRows(schema, ret.toArray.toSeq)
  }


  private def aggregatableColumns: Seq[Expression] = {
    df.schema.fields
      .filter(f => f.dataType.isInstanceOf[NumericType])
      .map { n =>
        df.queryExecution.analyzed.resolveQuoted(n.name,
          df.sparkSession.sessionState.analyzer.resolver).get
      }
  }

  /** A convenient function to wrap a logical plan and produce a DataFrame. */
  @inline private def withPlan(logicalPlan: => LogicalPlan): DataFrame = {
    Dataset.ofRows(df.sparkSession, logicalPlan)
  }
}
