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

object Pos1TestApp extends App with LazyLogging {

  private val ObservableCount = 3
  private val OutputCount = 1

  private val ArgumentDefaults = Map(
    "model" -> "",
    "dataset" -> "",
    "numSamples" -> "100")

  require(args.length >= 2, "Missing argument")

  execute()

  private def execute() {
    logger.info("Start")

    val kvArgs = Arguments(args, ArgumentDefaults)

    val modelFile = kvArgs("model")
    require(!modelFile.isEmpty, "Missing model")

    val testPrefix = kvArgs("dataset")
    require(!testPrefix.isEmpty, "Missing dataset")

    val numSamples = kvArgs("numSamples").toInt

    logger.info("Data Loaded")

    val net = ModelSerializer.restoreMultiLayerNetwork(modelFile)
    logger.info(s"$modelFile Loaded")

    logger.info("Testing ...")

    val results = for {
      i <- 0 until numSamples
    } yield {
      val featuresFile = s"${testPrefix}_features_$i.csv"
      val labelsFile = s"${testPrefix}_labels_$i.csv"
      computeErrorAllSteps(net, featuresFile, labelsFile)
    }
    val (errs, counts) = results.unzip
    val err = errs.sum
    val count = counts.sum
    val ratio = err / count

    logger.info(f"Square Error $err%f / $count%d = Mean Square $ratio%f")
    logger.info(s"Completed")
  }

  /** Step by step */
  private def computeErrorStepByStep(net: MultiLayerNetwork, featuresFile: String, labelsFile: String): (Double, Int) = {
    val features = csvread(new File(featuresFile))
    val labels = csvread(new File(labelsFile))
    val n = features.rows

    net.rnnClearPreviousState
    val result = DenseMatrix.zeros[Double](n, OutputCount)
    for {
      i <- 0 until n
    } {
      val in = Nd4j.zeros(1, ObservableCount)
      for {
        j <- 0 until ObservableCount
      } {
        in.putScalar(Array(0, j), features(i, j))
      }

      val out = net.rnnTimeStep(in)
      for {
        j <- 0 until OutputCount
      } {
        result(i, j) = out.getDouble(0, j)
      }
    }
    val diff = labels - result
    val d2 = diff :* diff
    val err = sum(d2)
    (err, n * OutputCount)
  }

  private def computeErrorAllSteps(net: MultiLayerNetwork, featuresFile: String, labelsFile: String): (Double, Int) = {
    val features = csvread(new File(featuresFile))
    val labels = csvread(new File(labelsFile))
    val n = features.rows

    net.rnnClearPreviousState
    val result = DenseMatrix.zeros[Double](n, OutputCount)
    for {
      i <- 0 until n
    } {
      val in = Nd4j.zeros(1, ObservableCount, 1)
      for {
        j <- 0 until ObservableCount
      } {
        in.putScalar(Array(0, j, 0), features(i, j))
      }

      val out = net.rnnTimeStep(in)
      //      val out = net.output(in)
      for {
        j <- 0 until OutputCount
      } {
        result(i, j) = out.getDouble(0, j, 0)
      }
    }
    val diff = labels - result
    val d2 = diff :* diff
    val err = sum(d2)
    (err, n * OutputCount)
  }

  private def computeErrorAllInOne(net: MultiLayerNetwork, featuresFile: String, labelsFile: String): (Double, Int) = {
    val features = csvread(new File(featuresFile))
    val labels = csvread(new File(labelsFile))
    val n = features.rows

    val in = Nd4j.zeros(1, ObservableCount, n)
    for {
      i <- 0 until n
      j <- 0 until ObservableCount
    } {
      in.putScalar(Array(0, j, i), features(i, j))
    }

    val out = net.output(in)
    val result = DenseMatrix.zeros[Double](n, OutputCount)
    for {
      i <- 0 until n
      j <- 0 until OutputCount
    } {
      result(i, j) = out.getDouble(0, j, i)
    }
    val diff = labels - result
    val d2 = diff :* diff
    val err = sum(d2)
    (err, n * OutputCount)
  }
}
