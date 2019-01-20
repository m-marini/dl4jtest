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

import org.nd4j.linalg.factory.Nd4j

import com.typesafe.scalalogging.LazyLogging

import breeze.stats.distributions.Rand
import breeze.stats.distributions.RandBasis
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import oracle.jrockit.jfr.DCmd

object SentenceSeqGenApp extends App with LazyLogging {
  private val TestEpisodeLength = 20
  private val SentenceCount = 10

  require(args.length >= 2, "Missing argument")

  execute()

  private def execute() {
    logger.info("Start")

    val modelName = args(0)
    val initial = args(1)
    val modelFile = new File(s"$modelName.zip")

    val dictionary = SentenceSeqTrainApp.loadDictionary(modelName)

    logger.info("Data Loaded")

    val inputLayer = dictionary.size
    val outputLayer = dictionary.size

    val net = ModelSerializer.restoreMultiLayerNetwork(modelFile)
    logger.info(s"$modelFile Loaded")

    logger.info("Testing ...")

    for {
      _ <- 1 to SentenceCount
    } {
      val sample = generateSentence(dictionary, net, TestEpisodeLength, initial)

      logger.info("Output:")
      logger.info(sample)
    }
  }

  def choiceIndex(pref: Seq[Double], rand: RandBasis): Int = {
    val (cum, _) = pref.foldLeft((Seq[Double](), 0.0)) {
      case ((acc, s), pref) =>
        val s1 = s + pref
        (acc :+ s1, s1)
    }
    val x = rand.uniform.sample
    cum.zipWithIndex.
      find(x < _._1).
      map(_._2).
      getOrElse(pref.length - 1)
  }

  def generateSentence(dictionary: Seq[String],
    net: MultiLayerNetwork,
    size: Int,
    initial: String,
    rand: RandBasis = Rand): String = {

    val state0 = dictionary.indexOf(initial)
    require(state0 >= 0, s"[$initial] not found in dictionary")

    val wordCount = dictionary.length
    net.rnnClearPreviousState
    //      val state0 = randB.randInt(dictionary.length).sample
    val (indexes, _) = (1 to size).foldLeft((Seq[Int](), state0)) {
      case ((sentence, state), _) =>

        val in = Nd4j.zeros(1, wordCount, 1)
        in.putScalar(Array(0, state, 0), 1.0)

        val out = net.rnnTimeStep(in)

        val pref = for { i <- 0 until wordCount } yield out.getDouble(0, i, 0)

        val next = choiceIndex(pref, rand)
        (sentence :+ state, next)
    }
    indexes.map(dictionary).mkString(" ")
  }
}
