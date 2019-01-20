// Copyright (c) 2016 Marco Marini, marco.marini@mmarini.org
//
// Licensed under the MIT License (MIT);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/MIT
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

package org.mmarini.lstm

import java.io.File

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j

import com.typesafe.scalalogging.LazyLogging

import breeze.linalg.DenseMatrix
import breeze.linalg.csvread
import breeze.linalg.sum
import javafx.beans.Observable
import breeze.stats.distributions.Rand

object Pos1DummyApp extends App with LazyLogging {

  private val OutputCount = 1

  private val ArgumentDefaults = Map(
    "dataset" -> "",
    "numSamples" -> "100",
    "mean" -> "0",
    "std" -> "1")

  execute()

  private def execute() {
    logger.info("Start")

    val kvArgs = Arguments(args, ArgumentDefaults)

    val testPrefix = kvArgs("dataset")
    require(!testPrefix.isEmpty, "Missing dataset")

    val numSamples = kvArgs("numSamples").toInt
    val mean = kvArgs("mean").toDouble
    val std = kvArgs("std").toDouble

    logger.info("Data Loaded")

    logger.info("Testing ...")

    val results = for {
      i <- 0 until numSamples
    } yield {
      val labelsFile = s"${testPrefix}_labels_$i.csv"
      computeError1(labelsFile, mean, std)
    }
    val (errs, counts) = results.unzip
    val err = errs.sum
    val count = counts.sum
    val ratio = err / count

    logger.info(f"Square error $err%f / $count%d = RMS $ratio%f")
    logger.info(s"Completed")
  }

  private def computeError1(labelsFile: String, mean: Double, std: Double): (Double, Int) = {
    val labels = csvread(new File(labelsFile))
    val n = labels.rows

    val result = DenseMatrix.zeros[Double](n, OutputCount)
    for {
      i <- 0 until n
      j <- 0 until OutputCount
    } {
      result(i, j) = Rand.gaussian.sample * std + mean
    }
    val diff = labels - result
    val d2 = diff :* diff
    val err = sum(d2)
    (err, n * OutputCount)
  }
}
