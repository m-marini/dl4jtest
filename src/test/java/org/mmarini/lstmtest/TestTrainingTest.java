package org.mmarini.lstmtest;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;

class TestTrainingTest {

	private static final int MINI_BATCH_SIZE = 10;
	private static final int NUM_LABELS = 1;
	private static final boolean REGRESSION = true;
	private static final String SAMPLES_FILE = "src/test/resources/datatest/sample_%d.csv";
	private static final String INPUTS_FILE = "src/test/resources/datatest/features_%d.csv";
	private static final int MIN_INPUTS_FILE_IDX = 0;
	private static final int MAX_INPUTS_FILE_IDX = 9;
	private static final String LABELS_FILE = "src/test/resources/datatest/labels_%d.csv";
	private static final int MIN_LABELS_FILE_IDX = 0;
	private static final int MAX_LABELS_FILE_IDX = 9;
	private static final int NUM_INPUTS_COLUMN = 1;
	private static final int NUM_HIDDEN_UNITS = 1;

	DataSetIterator createData() {
		final double[][][] featuresAry = new double[][][] { { { 0.5, 0.2, 0.5 } }, { { 0.5, 1.0, 0.0 } } };
		final double[] featuresData = ArrayUtil.flattenDoubleArray(featuresAry);
		final int[] featuresShape = new int[] { 2, 1, 3 };
		final INDArray features = Nd4j.create(featuresData, featuresShape, 'c');

		final double[][][] labelsAry = new double[][][] { { { 1.0, -1.0, 1.0 }, { 1.0, -1.0, -1.0 } } };
		final double[] labelsData = ArrayUtil.flattenDoubleArray(labelsAry);
		final int[] labelsShape = new int[] { 2, 1, 3 };
		final INDArray labels = Nd4j.create(labelsData, labelsShape, 'c');

		final INDArrayDataSetIterator iter = new INDArrayDataSetIterator(
				Arrays.asList(new Pair<INDArray, INDArray>(features, labels)), 2);
		System.out.println(iter.inputColumns());
		return iter;
	}

	private String file(String template) {
		return new File(".", template).getAbsolutePath();
	}

	@Test
	void testBuild() throws IOException, InterruptedException {
		final SingleFileTimeSeriesDataReader reader = new SingleFileTimeSeriesDataReader(file(SAMPLES_FILE),
				MIN_INPUTS_FILE_IDX, MAX_INPUTS_FILE_IDX, NUM_INPUTS_COLUMN, NUM_LABELS, MINI_BATCH_SIZE, REGRESSION);
//		final TimeSeriesDataReader reader = new TimeSeriesDataReader(MINI_BATCH_SIZE, NUM_LABELS, REGRESSION,
//				inputFile(), MIN_INPUTS_FILE_IDX, MAX_INPUTS_FILE_IDX, labelFile(), MIN_LABELS_FILE_IDX,
//				MAX_LABELS_FILE_IDX);

		final DataSetIterator data = reader.apply();
//		final DataSetIterator data = createData();

		assertThat(data.inputColumns(), equalTo(NUM_INPUTS_COLUMN));
		assertThat(data.totalOutcomes(), equalTo(NUM_LABELS));

		final TestConfBuilder builder = new TestConfBuilder(NUM_INPUTS_COLUMN, NUM_LABELS, NUM_HIDDEN_UNITS);
		final MultiLayerConfiguration conf = builder.build();
		final MultiLayerNetwork net = new MultiLayerNetwork(conf);
		assertNotNull(net);
		net.init();
		net.fit(data);
	}

}
