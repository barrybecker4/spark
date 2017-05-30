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

package org.apache.spark.sql.execution.datasources.csv

import java.math.BigDecimal
import java.sql.Timestamp
import java.text.NumberFormat
import java.util.Locale

import scala.util.{Failure, Success, Try}
import scala.util.control.Exception._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.analysis.TypeCoercion
import org.apache.spark.sql.catalyst.util.DateTimeUtils
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

private[csv] object CSVInferSchema {

  /**
   * Similar to the JSON schema inference
   *     1. Infer type of each row
   *     2. Merge row types to find common type
   *     3. Replace any null types with string type
   */
  def infer(
      tokenRdd: RDD[Array[String]],
      header: Array[String],
      options: CSVOptions): StructType = {
    val startType: Array[DataType] = Array.fill[DataType](header.length)(NullType)
    val rootTypes: Array[DataType] =
      tokenRdd.aggregate(startType)(inferRowType(options), mergeRowTypes)

    val structFields = header.zip(rootTypes).map { case (thisHeader, rootType) =>
      val dType = rootType match {
        case _: NullType => StringType
        case other => other
      }
      StructField(thisHeader, dType, nullable = true)
    }

    StructType(structFields)
  }

  private def inferRowType(options: CSVOptions)
      (rowSoFar: Array[DataType], next: Array[String]): Array[DataType] = {
    var i = 0
    while (i < math.min(rowSoFar.length, next.length)) {  // May have columns on right missing.
      rowSoFar(i) = inferField(rowSoFar(i), next(i), options)
      i+=1
    }
    rowSoFar
  }

  def mergeRowTypes(first: Array[DataType], second: Array[DataType]): Array[DataType] = {
    first.zipAll(second, NullType, NullType).map { case (a, b) =>
      findTightestCommonType(a, b).getOrElse(NullType)
    }
  }

  /**
   * Infer type of string field. Given known type Double, and a string "1", there is no
   * point checking if it is an Int, as the final type must be Double or higher.
   */
  def inferField(typeSoFar: DataType, field: String, options: CSVOptions): DataType = {
    if (field == null || field.isEmpty || options.nullValues.contains(field)) {
      typeSoFar
    } else {
      typeSoFar match {
        case NullType => tryParseInteger(field, options)
        case IntegerType => tryParseInteger(field, options)
        case LongType => tryParseLong(field, options)
        case _: DecimalType =>
          // DecimalTypes have different precisions and scales, so we try to find the common type.
          findTightestCommonType(typeSoFar, tryParseDecimal(field, options)).getOrElse(StringType)
        case DoubleType => tryParseDouble(field, options)
        case TimestampType => tryParseTimestamp(field, options)
        case BooleanType => tryParseBoolean(field, options)
        case StringType => StringType
        case other: DataType =>
          throw new UnsupportedOperationException(s"Unexpected data type $other")
      }
    }
  }

  private def tryParseInteger(field: String, options: CSVOptions): DataType = {
    if ((allCatch opt field.toInt).isDefined) {
      IntegerType
    } else {
      tryParseLong(field, options)
    }
  }

  private def tryParseLong(field: String, options: CSVOptions): DataType = {
    if ((allCatch opt field.toLong).isDefined) {
      LongType
    } else {
      tryParseDecimal(field, options)
    }
  }

  private def tryParseDecimal(field: String, options: CSVOptions): DataType = {
    val decimalTry = allCatch opt {
      // `BigDecimal` conversion can fail when the `field` is not a form of number.
      val bigDecimal = new BigDecimal(field)
      // Because many other formats do not support decimal, it reduces the cases for
      // decimals by disallowing values having scale (eg. `1.1`).
      if (bigDecimal.scale <= 0) {
        // `DecimalType` conversion can fail when
        //   1. The precision is bigger than 38.
        //   2. scale is bigger than precision.
        DecimalType(bigDecimal.precision, bigDecimal.scale)
      } else {
        tryParseDouble(field, options)
      }
    }
    decimalTry.getOrElse(tryParseDouble(field, options))
  }

  private def tryParseDouble(field: String, options: CSVOptions): DataType = {
    if ((allCatch opt field.toDouble).isDefined) {
      DoubleType
    } else {
      tryParseTimestamp(field, options)
    }
  }

  private def tryParseTimestamp(field: String, options: CSVOptions): DataType = {
    // This case infers a custom `dataFormat` is set.
    if ((allCatch opt options.timestampFormat.parse(field)).isDefined) {
      TimestampType
    } else if ((allCatch opt DateTimeUtils.stringToTime(field)).isDefined) {
      // We keep this for backwords competibility.
      TimestampType
    } else {
      tryParseBoolean(field, options)
    }
  }

  private def tryParseBoolean(field: String, options: CSVOptions): DataType = {
    if ((allCatch opt field.toBoolean).isDefined) {
      BooleanType
    } else {
      stringType()
    }
  }

  // Defining a function to return the StringType constant is necessary in order to work around
  // a Scala compiler issue which leads to runtime incompatibilities with certain Spark versions;
  // see issue #128 for more details.
  private def stringType(): DataType = {
    StringType
  }

  private val numericPrecedence: IndexedSeq[DataType] = TypeCoercion.numericPrecedence

  /**
   * Copied from internal Spark api
   * [[org.apache.spark.sql.catalyst.analysis.TypeCoercion]]
   */
  val findTightestCommonType: (DataType, DataType) => Option[DataType] = {
    case (t1, t2) if t1 == t2 => Some(t1)
    case (NullType, t1) => Some(t1)
    case (t1, NullType) => Some(t1)
    case (StringType, t2) => Some(StringType)
    case (t1, StringType) => Some(StringType)

    // Promote numeric types to the highest of the two and all numeric types to unlimited decimal
    case (t1, t2) if Seq(t1, t2).forall(numericPrecedence.contains) =>
      val index = numericPrecedence.lastIndexWhere(t => t == t1 || t == t2)
      Some(numericPrecedence(index))

    // These two cases below deal with when `DecimalType` is larger than `IntegralType`.
    case (t1: IntegralType, t2: DecimalType) if t2.isWiderThan(t1) =>
      Some(t2)
    case (t1: DecimalType, t2: IntegralType) if t1.isWiderThan(t2) =>
      Some(t1)

    // These two cases below deal with when `IntegralType` is larger than `DecimalType`.
    case (t1: IntegralType, t2: DecimalType) =>
      findTightestCommonType(DecimalType.forType(t1), t2)
    case (t1: DecimalType, t2: IntegralType) =>
      findTightestCommonType(t1, DecimalType.forType(t2))

    // Double support larger range than fixed decimal, DecimalType.Maximum should be enough
    // in most case, also have better precision.
    case (DoubleType, _: DecimalType) | (_: DecimalType, DoubleType) =>
      Some(DoubleType)

    case (t1: DecimalType, t2: DecimalType) =>
      val scale = math.max(t1.scale, t2.scale)
      val range = math.max(t1.precision - t1.scale, t2.precision - t2.scale)
      if (range + scale > 38) {
        // DecimalType can't support precision > 38
        Some(DoubleType)
      } else {
        Some(DecimalType(range + scale, scale))
      }

    case _ => None
  }
}

private[csv] object CSVTypeCast {

  private def castToByte(datum: String, options: CSVOptions): Option[Byte] = {
    if (options.permissive) {
      Try(datum.toByte) match {
        case Success(i) => Some(i)
        case Failure(_) => Try(datum.toDouble) match {
          case Success(d) => if (d.isValidByte) Some(d.toByte) else None
          case Failure(_) => None
        }
      }
    } else {
      Some(datum.toByte)
    }
  }

  private def castToShort(datum: String, options: CSVOptions): Option[Short] = {
    if (options.permissive) {
      Try(datum.toShort) match {
        case Success(i) => Some(i)
        case Failure(_) => Try(datum.toDouble) match {
          case Success(d) => if (d.isValidShort) Some(d.toShort) else None
          case Failure(_) => None
        }
      }
    } else {
      Some(datum.toShort)
    }
  }

  private def castToInteger(datum: String, options: CSVOptions): Option[Integer] = {
    if (options.permissive) {
      Try(datum.toInt) match {
        case Success(i) => Some(i)
        case Failure(_) => Try(datum.toDouble) match {
          case Success(d) => if (d.isValidInt) Some(d.toInt) else None
          case Failure(_) => None
        }
      }
    } else {
      Some(datum.toInt)
    }
  }

  private def castToLong(datum: String, options: CSVOptions): Option[Long] = {
    if (options.permissive) {
      Try(datum.toLong) match {
        case Success(l) => Some(l)
        case Failure(_) => Try(datum.toDouble) match {
          case Success(d) => Try(d.toLong) match {
            case Success(l) => Some(l)
            case Failure(_) => None
          }
          case Failure(_) => None
        }
      }
    } else {
      Some(datum.toLong)
    }
  }

  private def parseNumber(datum: String)
    = NumberFormat.getInstance(Locale.US).parse(datum)

  private def castToFloat(datum: String, options: CSVOptions): Option[Float] = {
    datum match {
      case options.nanValue => Some(Float.NaN)
      case options.negativeInf => Some(Float.NegativeInfinity)
      case options.positiveInf => Some(Float.PositiveInfinity)
      case _ => Try(datum.toFloat) match {
        case Success(f) => Some(f)
        case Failure(_) if options.permissive => Try(parseNumber(datum).floatValue()) match {
          case Success(f) => Some(f)
          case Failure(_) => None
        }
        case _ => Some(parseNumber(datum).floatValue)
      }
    }
  }

  private def castToDouble(datum: String, options: CSVOptions): Option[Double] = {
    datum match {
      case options.nanValue => Some(Double.NaN)
      case options.negativeInf => Some(Double.NegativeInfinity)
      case options.positiveInf => Some(Double.PositiveInfinity)
      case _ => Try(datum.toDouble) match {
        case Success(d) => Some(d)
        case Failure(_) if options.permissive => Try(parseNumber(datum).doubleValue()) match {
          case Success(d) => Some(d)
          case Failure(_) => None
        }
        case _ => Some(parseNumber(datum).doubleValue())
      }
    }
  }

  private def castToBoolean(datum: String, options: CSVOptions): Option[Boolean] = {
    if (options.permissive) {
      Try(datum.toBoolean) match {
        case Success(b) => Some(b)
        case Failure(_) => None
      }
    } else {
      Some(datum.toBoolean)
    }
  }

  private def stringToTime(datum: String) = DateTimeUtils.stringToTime(datum).getTime

  private def castToTimestamp(datum: String, options: CSVOptions): Option[Long] = {
    // This one will lose microseconds parts.
    // See https://issues.apache.org/jira/browse/SPARK-10681.
    Try(options.timestampFormat.parse(datum).getTime * 1000L) match {
      case Success(time) => Some(time)
      // If it fails to parse, then tries the way used in 2.0 and 1.x for backwards
      // compatibility.
      case Failure(_) if options.permissive => Try(stringToTime(datum) * 1000L) match {
        case Success(time) => Some(time)
        case Failure(_) => None
      }
      case _ => Some(stringToTime(datum) * 1000L)
    }
  }

  private def castToDate(datum: String, options: CSVOptions): Option[Integer] = {
    // This one will lose microseconds parts.
    // See https://issues.apache.org/jira/browse/SPARK-10681.x
    Try(DateTimeUtils.millisToDays(options.dateFormat.parse(datum).getTime)) match {
      case Success(time) => Some(time)
      // If it fails to parse, then tries the way used in 2.0 and 1.x for backwards
      // compatibility.
      case Failure(_) if options.permissive =>
        Try(DateTimeUtils.millisToDays(stringToTime(datum))) match {
          case Success(time) => Some(time)
          case Failure(_) => None
        }
      case _ => Some(DateTimeUtils.millisToDays(stringToTime(datum)))
    }
  }

  private def stringToDecimal(datum: String, dt: DecimalType): Decimal = {
      val value = new BigDecimal(datum.replaceAll(",", ""))
      Decimal(value, dt.precision, dt.scale)
  }

  private def castToDecimal(
    datum: String, dt: DecimalType, options: CSVOptions): Option[Decimal] = {
    if (options.permissive) {
      Try {
        stringToDecimal(datum, dt)
      } match {
        case Success(d) => Some(d)
        case Failure(_) => None
      }
    } else {
      Some(stringToDecimal(datum, dt))
    }
  }

  private def castToUTF8String(datum: String, options: CSVOptions): Option[UTF8String] = {
    if (options.permissive) {
      Try(UTF8String.fromString(datum)) match {
        case Success(s) => Some(s)
        case Failure(_) => None
      }
    } else {
      Some(UTF8String.fromString(datum))
    }
  }

  /**
   * Casts given string datum to specified type.
   * Currently we do not support complex types (ArrayType, MapType, StructType).
   *
   * For string types, this is simply the datum. For other types.
   * For other nullable types, returns null if it is null or equals to the value specified
   * in `nullValue` option.
   *
   * @param datum string value
   * @param name field name in schema.
   * @param castType data type to cast `datum` into.
   * @param nullable nullability for the field.
   * @param options CSV options.
   */
  def castTo(
      datum: String,
      name: String,
      castType: DataType,
      nullable: Boolean = true,
      options: CSVOptions = CSVOptions()): Any = {

    // datum can be null if the number of fields found is less than the length of the schema
    if (options.nullValues.contains(datum) || null == datum) {
      if (!nullable) {
        throw new RuntimeException(s"null value found but field $name is not nullable.")
      }
      null
    } else {
      val value = castType match {
        case _: ByteType => castToByte(datum, options)
        case _: ShortType => castToShort(datum, options)
        case _: IntegerType => castToInteger(datum, options)
        case _: LongType => castToLong(datum, options)
        case _: FloatType => castToFloat(datum, options)
        case _: DoubleType => castToDouble(datum, options)
        case _: BooleanType => castToBoolean(datum, options)
        case dt: DecimalType => castToDecimal(datum, dt, options)
        case _: TimestampType => castToTimestamp(datum, options)
        case _: DateType => castToDate(datum, options)
        case _: StringType => castToUTF8String(datum, options)
        case udt: UserDefinedType[_] => Some(castTo(datum, name, udt.sqlType, nullable, options))
        case _ => throw new RuntimeException(s"Unsupported type: ${castType.typeName}")
      }

      value.getOrElse(null)
    }
  }

  /**
   * Helper method that converts string representation of a character to actual character.
   * It handles some Java escaped strings and throws exception if given string is longer than one
   * character.
   */
  @throws[IllegalArgumentException]
  def toChar(str: String): Char = {
    if (str.charAt(0) == '\\') {
      str.charAt(1)
      match {
        case 't' => '\t'
        case 'r' => '\r'
        case 'b' => '\b'
        case 'f' => '\f'
        case '\"' => '\"' // In case user changes quote char and uses \" as delimiter in options
        case '\'' => '\''
        case 'u' if str == """\u0000""" => '\u0000'
        case _ =>
          throw new IllegalArgumentException(s"Unsupported special character for delimiter: $str")
      }
    } else if (str.length == 1) {
      str.charAt(0)
    } else {
      throw new IllegalArgumentException(s"Delimiter cannot be more than one character: $str")
    }
  }
}
