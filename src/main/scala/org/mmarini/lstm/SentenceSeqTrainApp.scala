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
import java.nio.charset.Charset
import java.nio.file.Files

import scala.collection.JavaConversions.asScalaBuffer
import scala.math.min

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import com.typesafe.scalalogging.LazyLogging

import breeze.stats.distributions.Rand
import breeze.stats.distributions.RandBasis
import org.canova.api.records.reader.impl.CSVSequenceRecordReader
import org.canova.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.util.ModelSerializer

object SentenceSeqTrainApp extends App with LazyLogging {

  private val HiddenLayerCont = 1
  private val HiddenLayerSize = 50
  private val IterationCount = 100
  private val Seed = 123
  private val SampleCount = 8
  private val TestSentenceSize = 20
  private val NumEpochs = 10
  //  private val BPTTTruncated = 10

  override def main(args: Array[String]) {
    super.main(args)
    require(args.length >= 1, "Missing argument")

    val modelName = args(0)

    val dictionary = loadDictionary(modelName)

    val inputLayer = dictionary.size
    val outputLayer = dictionary.size

    val net = {
      val net = createNet(inputLayer, outputLayer)
      logger.info("Network init")
      net
    }

    net.setListeners(new ScoreIterationListener(1))

    val data = loadTrainSet(modelName, SampleCount, SampleCount, outputLayer)
    for { _ <- 1 to NumEpochs } {
      net fit data
      data.reset

      val text = SentenceSeqGenApp.generateSentence(dictionary, net, TestSentenceSize, "nel")
      logger.info(s"Try: $text")
    }

    val modelFile = new File(s"$modelName.zip")

    ModelSerializer.writeModel(net, modelFile, true);

    logger.info("Trained")
  }

  /**
   *
   */
  private def createNet(inputLayerSize: Int, outputLayerSize: Int) = {

    // some common parameters
    val builder = new NeuralNetConfiguration.Builder()
      .iterations(IterationCount)
      .learningRate(0.001)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .seed(Seed)
      .biasInit(0)
      .miniBatch(false)
      .updater(Updater.RMSPROP)
      .weightInit(WeightInit.XAVIER)

    val listBuilder = builder.list(HiddenLayerCont + 1)

    // first difference, for rnns we need to use GravesLSTM.Builder
    for {
      i <- 0 until HiddenLayerCont
    } {
      val hiddenLayerBuilder = new GravesLSTM.Builder()
        .nIn(if (i == 0) inputLayerSize else HiddenLayerSize)
        .nOut(HiddenLayerSize)
        // adopted activation function from GravesLSTMCharModellingExample
        // seems to work well with RNNs
        .activation("tanh")
      listBuilder.layer(i, hiddenLayerBuilder.build())
    }

    // we need to use RnnOutputLayer for our RNN
    // softmax normalizes the output neurons, the sum of all outputs is 1
    // this is required for our sampleFromDistribution-function
    val outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT)
      .activation("softmax")
      .nIn(HiddenLayerSize)
      .nOut(outputLayerSize)
    listBuilder.layer(HiddenLayerCont, outputLayerBuilder.build())

    // finish builder
    listBuilder.pretrain(false)
      .backprop(true)

    // create network
    val conf = listBuilder.build()
    val net = new MultiLayerNetwork(conf)
    net.init()
    net
  }

  /**
   *
   */
  def loadTrainSet(file: String, samples: Int, miniBatchSize: Int, labelsSize: Int): DataSetIterator = {
    val featuresName = s"${file}_features_%d.csv"
    val labelsName = s"${file}_labels_%d.csv"
    val featureReader = new CSVSequenceRecordReader(0, ",")
    featureReader.initialize(new NumberedFileInputSplit(featuresName, 0, samples - 1))
    val labelReader = new CSVSequenceRecordReader(0, ",")
    labelReader.initialize(new NumberedFileInputSplit(labelsName, 0, samples - 1))
    val dsi = new SequenceRecordReaderDataSetIterator(
      featureReader,
      labelReader,
      miniBatchSize,
      labelsSize,
      true,
      SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
    dsi
  }

  /**
   *
   */
  private def createEpisode(indexes: Seq[Int], idx: Int, size: Int): Seq[(Int, Int)] = {
    val in = indexes.drop(idx).take(size)
    val out = indexes.drop(idx + 1).take(size)
    in zip out
  }

  /**
   *
   */
  def loadDictionary(name: String): Seq[String] =
    Files.readAllLines(new File(s"$name.dic").toPath(), Charset.defaultCharset()).toSeq
}
