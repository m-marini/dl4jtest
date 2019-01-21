/**
 *
 */
package org.mmarini.lstmtest;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;

/**
 * @author mmarini
 *
 */
public class TestConfBuilder {

	private final int noInputUnits;
	private final int noOutputUnits;
	private final int noLstmUnits;

	/**
	 *
	 * @param noInputUnits
	 * @param noOutputUnits
	 * @param noLstmUnits
	 */
	public TestConfBuilder(final int noInputUnits, final int noOutputUnits, final int noLstmUnits) {
		super();
		this.noInputUnits = noInputUnits;
		this.noOutputUnits = noOutputUnits;
		this.noLstmUnits = noLstmUnits;
	}

	/**
	 *
	 * @return
	 */
	public MultiLayerConfiguration build() {
		final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		final LSTM lstmLayer = new LSTM.Builder().units(noLstmUnits).nIn(noInputUnits).activation(Activation.TANH)
				.build();
		final RnnOutputLayer outLayer = new RnnOutputLayer.Builder(LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
				.activation(Activation.IDENTITY).nOut(noOutputUnits).nIn(noLstmUnits).build();
		final MultiLayerConfiguration conf = builder.list(lstmLayer, outLayer).build();
		return conf;
	}
}
