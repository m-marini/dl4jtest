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
import org.scalatest.Matchers
import org.scalatest.PropSpec
import org.scalatest.prop.PropertyChecks
import org.scalacheck.Gen

class TestDataset extends PropSpec with PropertyChecks with Matchers {
  private val MiniBatchSize = 5
  private val NumPossibleLabels = 3
  private val SkippedHeaderLines = 0
  private val MinIndex = 0
  private val MaxIndex = 9
  private val RegressionFlag = false

  property("Test Dataset") {
    forAll(
      (Gen.const("data/feature"), "features"),
      (Gen.const("data/label"), "labels")) {
        (features, labels) =>
          {
            val featureReader = new CSVSequenceRecordReader(SkippedHeaderLines, ",")
            featureReader.initialize(new NumberedFileInputSplit(features + "_%d.csv", MinIndex, MaxIndex))
            val labelReader = new CSVSequenceRecordReader(SkippedHeaderLines, ",")
            labelReader.initialize(new NumberedFileInputSplit(labels + "_%d.csv", MinIndex, MaxIndex))
            val dsi = new SequenceRecordReaderDataSetIterator(
              featureReader,
              labelReader,
              MiniBatchSize,
              NumPossibleLabels,
              RegressionFlag,
              SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
            dsi.reset
            dsi.batch should be(MiniBatchSize)

            var ct = 0
            while (dsi.hasNext) {
              dsi.next
              ct += 1;
            }
            ct should be(2)

            dsi.reset
            dsi.hasNext should be(true)
            val ds = dsi.next

            /*
             * 6x3
             * 6x3
             * 7x3
             * 7x3
             * 5x3
             */
            val f = ds.getFeatures
            f.length shouldBe (35 * 3)

            f.rank shouldBe (3)

            f.size(0) shouldBe (MiniBatchSize)
            f.size(1) shouldBe (3)
            f.size(2) shouldBe (7) // Max size

            // f.getDouble(i, j, k) : i-th sample, j-th feature, k-th time step
            f.getDouble(0, 0, 0) shouldBe -0.4700663926992381 +- 1e-6
            f.getDouble(0, 0, 5) shouldBe -2.574803255857511 +- 1e-6

            f.getDouble(0, 2, 0) shouldBe 0.8441587211845136 +- 1e-6
            f.getDouble(0, 2, 5) shouldBe -0.05883052048675538 +- 1e-6

            val fm = ds.getFeaturesMaskArray
            fm.length shouldBe 35
            fm.rank shouldBe 2

            fm.size(0) shouldBe MiniBatchSize
            fm.size(1) shouldBe 7

            fm.getDouble(0, 0) shouldBe 1.0
            fm.getDouble(0, 5) shouldBe 1.0
            fm.getDouble(0, 6) shouldBe 0.0

            fm.getDouble(4, 0) shouldBe 1.0
            fm.getDouble(4, 4) shouldBe 1.0
            fm.getDouble(4, 5) shouldBe 0.0

            val l = ds.getLabels
            l.length should be(35 * 3)
            l.rank should be(3)
            l.size(0) should be(MiniBatchSize)
            l.size(1) should be(3)
            l.size(2) should be(7) // Max size

            val lm = ds.getLabelsMaskArray
            lm.length should be(35)
          }
      }
  }
}
