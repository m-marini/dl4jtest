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
public class TimeSeriesDataReader {

	private final int miniBatchSize;
	private final int numPossibleLabels;
	private final boolean regression;
	private final String inputFilePattern;
	private final int maxInputsFileIdx;
	private final int minInputsFileIdx;
	private final String labelsFilePattern;
	private final int maxLabelsFileIdx;
	private final int minLabelsFileIdx;

	/**
	 * 
	 * @param miniBatchSize
	 * @param numPossibleLabels
	 * @param regression
	 * @param inputFilePattern
	 * @param minInputsFileIdx
	 * @param maxInputsFileIdx
	 * @param labelsFilePattern
	 * @param minLabelsFileIdx
	 * @param maxLabelsFileIdx
	 */
	public TimeSeriesDataReader(final int miniBatchSize, final int numPossibleLabels, final boolean regression,
			final String inputFilePattern, final int minInputsFileIdx, final int maxInputsFileIdx,
			final String labelsFilePattern, final int minLabelsFileIdx, final int maxLabelsFileIdx) {
		this.miniBatchSize = miniBatchSize;
		this.numPossibleLabels = numPossibleLabels;
		this.regression = regression;
		this.inputFilePattern = inputFilePattern;
		this.maxInputsFileIdx = maxInputsFileIdx;
		this.minInputsFileIdx = minInputsFileIdx;
		this.labelsFilePattern = labelsFilePattern;
		this.maxLabelsFileIdx = maxLabelsFileIdx;
		this.minLabelsFileIdx = minLabelsFileIdx;
	}

	/**
	 *
	 * @return
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public DataSetIterator apply() throws IOException, InterruptedException {
		final SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
		final SequenceRecordReader labelReader = new CSVSequenceRecordReader(0, ",");
		featureReader.initialize(new NumberedFileInputSplit(inputFilePattern, minInputsFileIdx, maxInputsFileIdx));
		labelReader.initialize(new NumberedFileInputSplit(labelsFilePattern, minLabelsFileIdx, maxLabelsFileIdx));
		final DataSetIterator iter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize,
				numPossibleLabels, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
		return iter;
	}
}
