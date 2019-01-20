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

import org.canova.api.records.reader.impl.CSVSequenceRecordReader
import org.canova.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import com.typesafe.scalalogging.LazyLogging

object Pos1TrainApp extends App with LazyLogging {

  private val ObservableCount = 3
  private val OutputCount = 1

  private val ArgumentDefaults = Map(
    "model" -> "",
    "dataset" -> "",
    "numSamples" -> "100",
    "numIterations" -> "1",
    "seed" -> "123",
    "numEpochs" -> "1",
    "batchSize" -> "5",
    "hiddens" -> "50")

  override def main(args: Array[String]) {
    super.main(args)
    val kvArgs = Arguments(args, ArgumentDefaults)

    val model = kvArgs("model")
    require(!model.isEmpty, "Missing model")

    val dataset = kvArgs("dataset")
    require(!dataset.isEmpty, "Missing dataset")

    val numSamples = kvArgs("numSamples").toInt
    val numIterations = kvArgs("numIterations").toInt
    val seed = kvArgs("seed").toInt
    val numEpochs = kvArgs("numEpochs").toInt
    val batchSize = kvArgs("batchSize").toInt
    val hiddens = kvArgs("hiddens").split(",").map(_.toInt)
    require(!hiddens.isEmpty, "Missing hiddens")

    val net = {
      val net = createNet(ObservableCount, OutputCount, hiddens, numIterations, seed)
      logger.info("Network init")
      net
    }

    net.setListeners(new ScoreIterationListener(1))
    //    net.setListeners(new HistogramIterationListener(1))

    val data = loadTrainSet(dataset, numSamples, batchSize, OutputCount)
    for { i <- 1 to numEpochs } {
      net fit data
      data.reset
      ModelSerializer.writeModel(net, model, true);
      logger.info(s"Epoch $i")
    }
    ModelSerializer.writeModel(net, model, true);

    logger.info("Trained")
  }

  /**
   *
   */
  private def createNet(inputLayerSize: Int,
    outputLayerSize: Int,
    hiddens: Seq[Int],
    numIterations: Int,
    seed: Int) = {

    // some common parameters
    val builder = new NeuralNetConfiguration.Builder()
      .iterations(numIterations)
      .learningRate(0.001)
      .regularization(true)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .seed(seed)
      .biasInit(0)
      .miniBatch(true)
      .updater(Updater.RMSPROP)
      .weightInit(WeightInit.XAVIER)

    val listBuilder = builder.list

    // first difference, for rnns we need to use GravesLSTM.Builder
    for {
      ((outNeuro, inNeuro), layer) <- hiddens zip (inputLayerSize +: hiddens.init) zipWithIndex
    } {
      val hiddenLayerBuilder = new GravesLSTM.Builder()
        .nIn(inNeuro)
        .nOut(outNeuro)
        // adopted activation function from GravesLSTMCharModellingExample
        // seems to work well with RNNs
        .activation("tanh")
      listBuilder.layer(layer, hiddenLayerBuilder.build())
    }

    // we need to use RnnOutputLayer for our RNN
    // softmax normalizes the output neurons, the sum of all outputs is 1
    // this is required for our sampleFromDistribution-function
    val outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MSE)
      .activation("identity")
      .nIn(hiddens.last)
      .nOut(outputLayerSize)
    listBuilder.layer(hiddens.length, outputLayerBuilder.build())

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
}
