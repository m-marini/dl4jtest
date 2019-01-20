package org.mmarini.lstmtest;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.io.File;
import java.io.IOException;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 *
 * @author us00852
 *
 */
class TimeSeriesDataReaderTest {

	private static final int TIME_SEQUENCE_LENGTH = 10;
	private static final int NUM_INPUTS_COLUMN = 1;
	private static final int MINI_BATCH_SIZE = 2;
	private static final int NUM_LABELS = 1;
	private static final boolean REGRESSION = true;
	private static final String INPUTS_FILE = "src/test/resources/datatest/features_%d.csv";
	private static final int MIN_INPUTS_FILE_IDX = 0;
	private static final int MAX_INPUTS_FILE_IDX = 9;
	private static final int NUM_SAMPLES = MAX_INPUTS_FILE_IDX - MIN_INPUTS_FILE_IDX + 1;
	private static final String LABELS_FILE = "src/test/resources/datatest/labels_%d.csv";
	private static final int MIN_LABELS_FILE_IDX = 0;
	private static final int MAX_LABELS_FILE_IDX = 9;

	private TimeSeriesDataReader reader;

	private String inputFile() {
		return new File(".", INPUTS_FILE).getAbsolutePath();
	}

	private String labelFile() {
		return new File(".", LABELS_FILE).getAbsolutePath();
	}

	@BeforeEach
	void setup() {
		reader = new TimeSeriesDataReader(MINI_BATCH_SIZE, NUM_LABELS, REGRESSION, inputFile(), MIN_INPUTS_FILE_IDX,
				MAX_INPUTS_FILE_IDX, labelFile(), MIN_LABELS_FILE_IDX, MAX_LABELS_FILE_IDX);

	}

	@Test
	void testApply() throws IOException, InterruptedException {
		final DataSetIterator iter = reader.apply();
		assertThat(iter.batch(), equalTo(MINI_BATCH_SIZE));
		assertThat(iter.totalOutcomes(), equalTo(NUM_LABELS));
		assertThat(iter.inputColumns(), equalTo(NUM_INPUTS_COLUMN));
	}

	@Test
	void testApplyAndIterate() throws IOException, InterruptedException {
		final DataSetIterator iter = reader.apply();

		assertThat(iter.hasNext(), equalTo(true));
		{
			final DataSet dataSet = iter.next();
			assertNotNull(dataSet);

			final INDArray features = dataSet.getFeatures();
			final INDArray featuresMask = dataSet.getFeaturesMaskArray();
			final INDArray labels = dataSet.getLabels();
			final INDArray labelsMask = dataSet.getLabelsMaskArray();

			assertNotNull(features);
			// Check for first sample
			assertThat(features.getDouble(new long[] { 0, 0, 0 }), closeTo(4.76337e-1, 1e-6));
			assertThat(features.getDouble(new long[] { 0, 0, 7 }), closeTo(4.76337e-1, 1e-6));
			// Check for padding
//			assertThat(features.getDouble(new long[] { 1, 0, 6 }), closeTo(0.0, 1e-6));
			// check for second sample
			assertThat(features.getDouble(new long[] { 1, 0, 0 }), closeTo(5.00488e-3, 1e-8));
			assertThat(features.getDouble(new long[] { 1, 0, 4 }), closeTo(5.00488e-3, 1e-6));
			// Check for padding
			assertThat(features.getDouble(new long[] { 1, 0, 5 }), closeTo(0.0, 1e-6));

			assertNotNull(featuresMask);
			// Check for first mask
			assertThat(featuresMask.getDouble(new long[] { 0, 0 }), closeTo(1.0, 1e-6));
			assertThat(featuresMask.getDouble(new long[] { 0, 7 }), closeTo(1.0, 1e-6));
			// Check for masking
//			assertThat(featuresMask.getDouble(new long[] { 0, 8 }), closeTo(0.0, 1e-6));
			// check for second sample
			assertThat(featuresMask.getDouble(new long[] { 1, 0 }), closeTo(1.0, 1e-6));
			assertThat(featuresMask.getDouble(new long[] { 1, 4 }), closeTo(1.0, 1e-6));
			// Check for masking
			assertThat(featuresMask.getDouble(new long[] { 1, 5 }), closeTo(0.0, 1e-6));

			assertThat(labels.rank(), equalTo(3));
			// Check for first sample
			assertThat(labels.getDouble(new long[] { 0, 0, 0 }), closeTo(1.0, 1e-6));
			assertThat(labels.getDouble(new long[] { 0, 0, 7 }), closeTo(1.0, 1e-6));
			// Check for padding
//			assertThat(labels.getDouble(new long[] { 0, 2, 8 }), closeTo(0.0, 1e-6));
			// check for second sample
			assertThat(labels.getDouble(new long[] { 1, 0, 0 }), closeTo(1.0, 1e-6));
			assertThat(labels.getDouble(new long[] { 1, 0, 4 }), closeTo(1.0, 1e-6));
			// Check for padding
			assertThat(labels.getDouble(new long[] { 1, 0, 5 }), closeTo(0.0, 1e-6));

			assertNotNull(labelsMask);
			// Check for first mask
			assertThat(labelsMask.getDouble(new long[] { 0, 0 }), closeTo(1.0, 1e-6));
			assertThat(labelsMask.getDouble(new long[] { 0, 7 }), closeTo(1.0, 1e-6));
			// Check for masking
//			assertThat(labelsMask.getDouble(new long[] { 0, 8 }), closeTo(0.0, 1e-6));
			// check for second sample
			assertThat(labelsMask.getDouble(new long[] { 1, 0 }), closeTo(1.0, 1e-6));
			assertThat(labelsMask.getDouble(new long[] { 1, 4 }), closeTo(1.0, 1e-6));
			// Check for masking
			assertThat(labelsMask.getDouble(new long[] { 1, 5 }), closeTo(0.0, 1e-6));
		}

		assertThat(iter.hasNext(), equalTo(true));
		{
			final DataSet dataSet = iter.next();
			assertNotNull(dataSet);

			final INDArray features = dataSet.getFeatures();
			final INDArray featuresMask = dataSet.getFeaturesMaskArray();
			final INDArray labels = dataSet.getLabels();
			final INDArray labelsMask = dataSet.getLabelsMaskArray();

			assertNotNull(features);
			// Check for third sample
			assertThat(features.getDouble(new long[] { 0, 0, 0 }), closeTo(9.470363e-1, 1e-6));
			assertThat(features.getDouble(new long[] { 0, 0, 9 }), closeTo(7.557169e-1, 1e-6));
			// Check for padding
//			assertThat(features.getDouble(new long[] { 0, 2, 7 }), closeTo(0.0, 1e-6));
			// check for fourth sample
			assertThat(features.getDouble(new long[] { 1, 0, 0 }), closeTo(5.39766e-1, 1e-6));
			assertThat(features.getDouble(new long[] { 1, 0, 9 }), closeTo(2.58672e-1, 1e-6));
			// Check for padding
//			assertThat(features.getDouble(new long[] { 1, 0, 6 }), closeTo(0.0, 1e-6));

			assertNull(featuresMask);
//			// Check for third mask
//			assertThat(featuresMask.getDouble(new long[] { 0, 0 }), closeTo(1.0, 1e-6));
//			assertThat(featuresMask.getDouble(new long[] { 0, 9 }), closeTo(1.0, 1e-6));
//			// check for fourth sample
//			assertThat(featuresMask.getDouble(new long[] { 1, 0 }), closeTo(1.0, 1e-6));
//			assertThat(featuresMask.getDouble(new long[] { 1, 5 }), closeTo(1.0, 1e-6));
//			assertThat(featuresMask.getDouble(new long[] { 1, 6 }), closeTo(0.0, 1e-6));

			assertThat(labels.rank(), equalTo(3));
			// Check for third sample
			assertThat(labels.getDouble(new long[] { 0, 0, 0 }), closeTo(1.0, 1e-6));
			assertThat(labels.getDouble(new long[] { 0, 0, 9 }), closeTo(-1.0, 1e-6));
			// check for forth sample
			assertThat(labels.getDouble(new long[] { 1, 0, 0 }), closeTo(1.0, 1e-6));
			assertThat(labels.getDouble(new long[] { 1, 0, 9 }), closeTo(-1.0, 1e-6));

			assertNull(labelsMask);
//			// Check for third mask
//			assertThat(labelsMask.getDouble(new long[] { 0, 0 }), closeTo(1.0, 1e-6));
//			assertThat(labelsMask.getDouble(new long[] { 0, 6 }), closeTo(1.0, 1e-6));
//			// Check for padding
//			assertThat(labelsMask.getDouble(new long[] { 0, 7 }), closeTo(0.0, 1e-6));
//			// check for forth sample
//			assertThat(labelsMask.getDouble(new long[] { 1, 0 }), closeTo(1.0, 1e-6));
//			assertThat(labelsMask.getDouble(new long[] { 1, 9 }), closeTo(1.0, 1e-6));
		}
	}

	@Test
	void testApplyAndIterateAll() throws IOException, InterruptedException {
		final DataSetIterator iter = reader.apply();
		int n = 0;
		while (iter.hasNext()) {

			final DataSet dataSet = iter.next();
			assertNotNull(dataSet);

			final INDArray features = dataSet.getFeatures();
			final INDArray featuresMask = dataSet.getFeaturesMaskArray();
			final INDArray labels = dataSet.getLabels();
			final INDArray labelsMask = dataSet.getLabelsMaskArray();

			assertThat(features.rank(), equalTo(3));
			assertThat(features.shape()[0], equalTo((long) MINI_BATCH_SIZE));
			assertThat(features.shape()[1], equalTo((long) NUM_INPUTS_COLUMN));

			if (featuresMask != null) {
				assertThat(featuresMask.rank(), equalTo(2));
				assertThat(featuresMask.shape()[0], equalTo((long) MINI_BATCH_SIZE));
			}

			assertThat(labels.rank(), equalTo(3));
			assertThat(labels.shape()[0], equalTo((long) MINI_BATCH_SIZE));
			assertThat(labels.shape()[1], equalTo((long) NUM_LABELS));

			if (labelsMask != null) {
				assertNotNull(labelsMask);
				assertThat(labelsMask.rank(), equalTo(2));
				assertThat(labelsMask.shape()[0], equalTo((long) MINI_BATCH_SIZE));
			}

			n++;
		}
		assertThat(n, equalTo(NUM_SAMPLES / MINI_BATCH_SIZE));
	}
}
