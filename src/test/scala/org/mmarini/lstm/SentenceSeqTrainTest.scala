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

class SentenceSeqTrainTest extends PropSpec with PropertyChecks with Matchers {
  private val SampleCount = 1
  private val LabelSize = 107

  property("Test Dataset") {
    forAll(
      (Gen.const("data/test/test"), "file")) {
        (file) =>
          {
            val dsi = SentenceSeqTrainApp.loadTrainSet(file, SampleCount, SampleCount, LabelSize)
            dsi.reset

            var ct = 0
            while (dsi.hasNext) {
              dsi.next()
              ct += 1;
            }
            ct shouldBe 1

            dsi.reset
            dsi.hasNext should be(true)
            val ds = dsi.next()

            val f = ds.getFeatures

            f.rank shouldBe 3
            f.size(0) shouldBe 1
            f.size(1) shouldBe 107
            f.size(2) shouldBe 6

            val l = ds.getLabels

            l.rank shouldBe 3
            l.size(0) shouldBe 1
            l.size(1) shouldBe 107
            l.size(2) shouldBe 6

            for {
              i <- 0 until 107
            } {
              val lv = l.getDouble(0, i, 0)
              val fv = f.getDouble(0, i, 1)
              lv shouldBe fv
            }
          }
      }
  }
}
