package org.mmarini.lstmtest;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

class TestTrainingTest {

	private static final int MINI_BATCH_SIZE = 10;
	private static final int NUM_LABELS = 1;
	private static final boolean REGRESSION = true;
	private static final String INPUTS_FILE = "src/test/resources/datatest/features_%d.csv";
	private static final int MIN_INPUTS_FILE_IDX = 0;
	private static final int MAX_INPUTS_FILE_IDX = 9;
	private static final String LABELS_FILE = "src/test/resources/datatest/labels_%d.csv";
	private static final int MIN_LABELS_FILE_IDX = 0;
	private static final int MAX_LABELS_FILE_IDX = 9;
	private static final int NUM_INPUTS_COLUMN = 1;
	private static final int NUM_HIDDEN_UNITS = 1;

	@Test
	void testBuild() throws IOException, InterruptedException {
		TimeSeriesDataReader reader = new TimeSeriesDataReader(MINI_BATCH_SIZE, NUM_LABELS, REGRESSION, inputFile(),
				MIN_INPUTS_FILE_IDX, MAX_INPUTS_FILE_IDX, labelFile(), MIN_LABELS_FILE_IDX, MAX_LABELS_FILE_IDX);
		DataSetIterator data = reader.apply();
		assertThat(data.totalOutcomes(), equalTo(NUM_LABELS));
		assertThat(data.inputColumns(), equalTo(NUM_INPUTS_COLUMN));

		TestConfBuilder builder = new TestConfBuilder(NUM_INPUTS_COLUMN, NUM_LABELS, NUM_HIDDEN_UNITS);
		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		assertNotNull(net);
		net.init();
		net.fit(data);
	}

	private String inputFile() {
		return new File(".", INPUTS_FILE).getAbsolutePath();
	}

	private String labelFile() {
		return new File(".", LABELS_FILE).getAbsolutePath();
	}

}
