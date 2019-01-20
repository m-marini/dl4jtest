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

import java.io.File
import java.io.IOException
import java.net.URL
import java.nio.charset.Charset
import java.nio.file.Files
import org.apache.commons.io.FileUtils
import com.typesafe.scalalogging.LazyLogging
import scala.collection.JavaConversions._

object SentenceSeqParseApp extends App with LazyLogging {

  require(args.length >= 2, "Wrong number of arguments")

  val (dictionary, text) = parseFile(args(0))
  saveData(args(1), dictionary, text)

  private def saveData(name: String, x: Seq[Seq[Seq[Int]]]) {
    for {
      (data, i) <- x.zipWithIndex
    } {
      val f = new File(s"${name}_$i.csv")
      f.getParentFile.mkdirs
      val content = data.map(_.mkString(","))
      Files.write(f.toPath, content)
    }
  }

  private def toFeatures(indexes: Seq[Seq[Int]], n: Int): Seq[Seq[Seq[Int]]] = {
    val zeroRow = Seq.fill(n)(0)
    val r = for {
      seq <- indexes
    } yield {
      for {
        i <- seq
      } yield {
        zeroRow.updated(i, 1)
      }
    }
    r
  }

  private def saveData(name: String, dictionary: Seq[String], indexes: Seq[Seq[Int]]) {
    val f = new File(s"$name.dic")
    f.getParentFile.mkdirs
    Files.write(f.toPath(), dictionary)
    logger.info(f"Written $name%s.dic with ${dictionary.size}%d words")
    val cv = for {
      row <- indexes
    } yield {
      row.map(String.valueOf).mkString(",")
    }

    saveData(s"${name}_features", toFeatures(indexes.map(_.init), dictionary.size))
    saveData(s"${name}_labels", toFeatures(indexes.map(_.tail), dictionary.size))
  }

  private def parseFile(url: String): (Seq[String], Seq[Seq[Int]]) = {
    val f = new File(url)
    require(f.exists(), s"Missing file $url")

    val text = for {
      line <- Files.readAllLines(f.toPath(), Charset.forName("UTF-8")).toSeq
    } yield {
      (for {
        word <- line.toLowerCase.split("\\W")
        if (!word.isEmpty)
      } yield word).toSeq
    }
    val dictionary = text.flatten.toSet.toSeq.sorted
    val map = dictionary.zipWithIndex.toMap
    val seq = for {
      line <- text
    } yield {
      for {
        word <- line
        idx <- map.get(word)
      } yield {
        idx
      }
    }
    logger.info(f"Loaded ${seq.size}%d lines in a ${dictionary.size}%d dictionary words")
    (dictionary, seq)
  }
}
