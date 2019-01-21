package org.mmarini.lstmtest;

import java.io.IOException;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 *
 */
public class SingleFileTimeSeriesDataReader {

	private final int miniBatchSize;
	private final int numPossibleLabels;
	private final boolean regression;
	private final String filePattern;
	private final int maxFileIdx;
	private final int minFileIdx;
	private final int numInputs;

	/**
	 * 
	 * @param filePattern
	 * @param minFileIdx
	 * @param maxFileIdx
	 * @param numInputs
	 * @param numPossibleLabels
	 * @param miniBatchSize
	 * @param regression
	 */
	public SingleFileTimeSeriesDataReader(final String filePattern, final int minFileIdx, final int maxFileIdx,
			final int numInputs, final int numPossibleLabels, final int miniBatchSize, final boolean regression) {
		this.miniBatchSize = miniBatchSize;
		this.numPossibleLabels = numPossibleLabels;
		this.regression = regression;
		this.filePattern = filePattern;
		this.maxFileIdx = maxFileIdx;
		this.minFileIdx = minFileIdx;
		this.numInputs = numInputs;
	}

	/**
	 *
	 * @return
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public DataSetIterator apply() throws IOException, InterruptedException {
		final SequenceRecordReader reader = new CSVSequenceRecordReader(0, ",");
		reader.initialize(new NumberedFileInputSplit(filePattern, minFileIdx, maxFileIdx));
		final DataSetIterator iter = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels,
				numInputs, regression);
		return iter;
	}
}
