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
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.conf.stepfunctions.DefaultStepFunction
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.weights.HistogramIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.nn.conf.stepfunctions.NegativeGradientStepFunction

object Pos2TrainApp extends App with LazyLogging {

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
    "hiddens" -> "",
    "lstms" -> "")

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
    val lstms = kvArgs("lstms").split(",").map(_.toInt)
    require(!hiddens.isEmpty, "Missing hiddens")

    val net = {
      val net = createNet(ObservableCount, OutputCount, hiddens zip lstms, numIterations, seed)
      logger.info("Network init")
      net
    }

    //    net.setListeners(new ScoreIterationListener(1))
    net.setListeners(new HistogramIterationListener(1))

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

  private def createLayer(builder: GraphBuilder,
    idLayer: Int,
    inputSize: Int,
    inputName: String,
    hiddenSize: Int,
    lstmSize: Int) = {

    val layerName = "layer_" + idLayer

    /* Creates merger vertex if necessary */
    val (hiddenName, lstmName) =
      if (hiddenSize > 0 && lstmSize > 0) {
        val hidden = "hiddenrnn_" + idLayer
        val lstm = "lstm_" + idLayer
        builder
          .addVertex(layerName, new MergeVertex(), hidden, lstm)
        (hidden, lstm)
      } else {
        (layerName, layerName)
      }

    /* Creates hidden vertex if necessary */
    if (hiddenSize > 0) {
      val name = "hidden_" + idLayer
      val hidden = new DenseLayer.Builder()
        .nIn(inputSize)
        .nOut(hiddenSize)
        .activation("tanh")
      builder
        .addLayer(name, hidden.build, new RnnToFeedForwardPreProcessor(), inputName)
        .addVertex(hiddenName, new PreprocessorVertex(new FeedForwardToRnnPreProcessor()), name)

    }

    /* Creates lstm verex if necessary */
    if (lstmSize > 0) {
      val lstm = new GravesLSTM.Builder()
        .nIn(inputSize)
        .nOut(lstmSize)
        .activation("tanh")
      builder
        .addLayer(lstmName, lstm.build, inputName)

    }
    builder
  }

  /**
   *
   */
  private def createNet(inputLayerSize: Int,
    outputLayerSize: Int,
    hiddens: Seq[(Int, Int)],
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
      .miniBatch(false)
      .updater(Updater.RMSPROP)
      .weightInit(WeightInit.XAVIER)
      .stepFunction(new NegativeGradientStepFunction())
      .graphBuilder()

    builder.addInputs("observables")

    val inputNames = "observables" +: (0 until hiddens.size - 1).map("layer_" + _)
    val inputSizes = inputLayerSize +: hiddens.init.map { case (a, b) => a + b }
    val layerParms = inputNames zip inputSizes zip hiddens zipWithIndex

    for {
      (((inName, inSize), (numHidden, numLstm)), layer) <- layerParms
    } {
      createLayer(builder, layer, inSize, inName, numHidden, numLstm)
    }

    // we need to use RnnOutputLayer for our RNN
    // softmax normalizes the output neurons, the sum of all outputs is 1
    // this is required for our sampleFromDistribution-function
    val lastSizes = hiddens.last
    val outputLayer = new RnnOutputLayer.Builder(LossFunction.MSE)
      .activation("identity")
      .nIn(lastSizes._1 + lastSizes._2)
      .nOut(outputLayerSize).build
    builder.addLayer("output", outputLayer, "layer_" + (hiddens.size - 1))
    // finish builder
    builder.pretrain(false)
      .backprop(true)

    // create network
    val conf = builder.setOutputs("output").build
    val net = new ComputationGraph(conf)
    net.init
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
