import java.io.{BufferedWriter, File, FileWriter}
import java.util.stream.Collectors

import breeze.linalg._
import com.cibo.evilplot.numeric.Point
import com.cibo.evilplot.plot.{LinePlot, _}
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.notebook._
import breeze.stats.mean
import java.io.PrintWriter

import org.scalacheck.Prop.Exception

import scala.annotation.tailrec
import scala.collection.immutable.ListMap
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object MapAnova2WithInters_vComp {

  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val (data, dataRaw, n1, n2) = dataProcessing()
    mainEffects(data, dataRaw, rng, n1, n2)
  }

  /**
   * Process data read from input file
   */
  def dataProcessing(): (Map[(Int, Int), List[Double]], (Array[Double], Array[Int], Array[Int]), Int, Int) = {
    val data = csvread(new File("/home/antonia/ResultsFromCloud/CompareRainier/040619/withInteractions/simulInter040619.csv"))
    val sampleSize = data.rows
    //println(sampleSize)
    val y = data(::, 0).toArray
    val alpha = data(::, 1).map(_.toInt)
    val beta = data(::, 2).map(_.toInt)
    val nj = alpha.toArray.distinct.length //the number of levels for the first variable
    val nk = beta.toArray.distinct.length //the number of levels for the second variable
    val l = alpha.length // the number of observations
    var dataList = List[(Int, Int)]()

    for (i <- 0 until l) {
      dataList = dataList :+ (alpha(i), beta(i))
    }
    //println(dataList)

    val dataMap = (dataList zip y).groupBy(_._1).map { case (k, v) => ((k._1 - 1, k._2 - 1), v.map(_._2)) } //Bring the data to the map format
    val dataRaw = (y.map(i => i-1), alpha.toArray.map(i => i-1), beta.toArray.map(i => i-1))
    //println(dataMap)
    (dataMap, dataRaw, nj, nk)

  }

  /**
   * Use Rainier for modelling the main effects only, without interactions
   */
  def mainEffects(dataMap: Map[(Int, Int), List[Double]], dataRaw: (Array[Double], Array[Int], Array[Int]), rngS: ScalaRNG, n1: Int, n2: Int): Unit = {

    // Implementation of sqrt for Real
    def sqrtR(x: Real): Real = {
      val lx = (Real(0.5) * x.log).exp
      lx
    }

    val obs = dataRaw._1
    val alpha = dataRaw._2
    val beta = dataRaw._3

    implicit val rng = rngS
    val n = dataMap.size //No of groups
    // Priors for the unknown parameters
    val mu = Normal(0, 100).latent

    // Sample tau, estimate sd to be used in sampling from Normal the effects for the 1st variable
    val tauE1RV = Gamma(1, 10000).latent
    val sdE1 = sqrtR(Real(1.0) / tauE1RV)

    // Sample tau, estimate sd to be used in sampling from Normal the effects for the 2nd variable
    val tauE2RV = Gamma(1, 10000).latent
    val sdE2 = sqrtR(Real(1.0) / tauE2RV)

    // Sample tau, estimate sd to be used in sampling from Normal for fitting the model
    val tauDRV = Gamma(1, 10000).latent
    val sdDR = sqrtR(Real(1.0) / tauDRV)

    // Sample tau, estimate sd to be used in sampling from Normal the interaction effects
    val tauInterRV = Gamma(1, 10000).latent
    val sdInter = sqrtR(Real(1.0) / tauInterRV)

    //Create a Vec[Real] for each effect
    val eff1p = Normal(0, sdE1).latentVec(n1)
    val eff2p = Normal(0, sdE2).latentVec(n2)
    //val interEffp = Normal(0, sdInter).latentVec(n1).map(i => Normal(0, sdInter).latentVec(n2))
    val interEffp = Normal(0, sdInter).latentVec(dataMap.keys.size)
    //println(interEffp.size)
    //print(dataMap.keys.size)

    val indSeqInters = dataMap.keys.toList.sortBy(_._1).sortBy(_._2).toIndexedSeq
    val inters = (alpha zip beta).map{ case(i,j) => indSeqInters.indexOf((i,j))}
    println(inters.toList)

    /**
     * Uses the imported data in format (y, alpha, beta)
     * 1. Creates a Vec[Real] for columns alpha & beta
     * 2. zips the vecs to one Vec[(Real, Real)] to map over
     * 3. maps over the Vec[(Real, Real)] to create a normal for each observation
     * 4. Builds the model
     *
     * Problem: Slow and wrong results
     * e.g. For HMC(100, 100, 50) time: 380505.616669ms
     * For HMC(1000, 1000, 100) 6788099.496656ms
     */
    def implementation2(): Model = {
      val eff1Vec = Vec.from(alpha) //1.
      val eff2Vec = Vec.from(beta) //1.
      val interEffVec = Vec.from(inters)
      val eff1VecZipEff2Vec = (eff1Vec zip eff2Vec zip interEffVec).map { case ((v1, v2), v3) => (v1, v2, v3) } //2.
      val modeleff1Vec = eff1VecZipEff2Vec.map{case (a,b,c) => {Normal(mu + eff1p(a) + eff2p(b) + interEffp(c), sdDR)}} //3.
      val vecModel = Model.observe(obs.toList, modeleff1Vec) //4.
      vecModel
    }


    def time[A](f: => A) = {
      val s = System.nanoTime
      val ret = f
      println("time: " + (System.nanoTime - s) / 1e6 + "ms")
      ret
    }

    println("sampling...")
    val vecModel = implementation2()
    val warmupIters = 10
    val samplesPerChain = 10
    val leapfrogSteps = 5
    val thinBy = 1
    val trace = time(vecModel.sample(HMC(warmupIters, samplesPerChain, leapfrogSteps)))
    println("sampling over...")
    val traceThinned = trace.thin(thinBy)
    val sampNo = samplesPerChain * 4 / thinBy

    val InterEffp2D = interEffp.toList.toArray.grouped(10).toArray
    val skatoul = traceThinned.predict(interEffp(1))
    println(s"skatoul" + skatoul.length)
    println(InterEffp2D)

    val postmuTaus = DenseMatrix(DenseVector(traceThinned.predict(mu).toArray), DenseVector(traceThinned.predict(tauDRV).toArray), DenseVector(traceThinned.predict(tauE1RV).toArray), DenseVector(traceThinned.predict(tauE2RV).toArray))


    def predictAllInters(myTrace: Trace, vc: Array[Array[Real]]) : Map[String, List[Double]] = {
      vc.toList.zipWithIndex.flatMap {
        case (e, i) => {
          val skatoules = e.toList.zipWithIndex.map {
            case (ee, j) => {
              s"effInter( $i, $j)" ->
                myTrace.predict(ee)
            }
          }
          skatoules
        }
      }.toMap
    }

    val dokimi= predictAllInters(traceThinned, InterEffp2D)

    def predictAll(myTrace: Trace, vc: Vec[Real]) : Map[String, List[Double]] = {
      vc.toList.zipWithIndex.map{
        case (e, i) => { "effA".concat(i.toString) -> myTrace.predict(e) }
      }.toMap
    }

    def printSamplesToCsv(filename: String, eff: Map[String, List[Double]]): Unit = {
      val pw = new PrintWriter(new File(filename))

      def flushLineToDisk(builder: StringBuilder): Unit = {
        builder.append("\n")
        pw.append(builder) //a. Write at each line
        pw.flush() //a.
        builder.clear()
      }

      val builder = new StringBuilder()

      val sortedMap = ListMap(eff.toSeq.sortBy(_._1): _*)

      //write titles to csv and export items as List[List[Double]]
      val listOfItems = sortedMap.map( mapentry => {
        builder.append(mapentry._1.concat(","))
        mapentry._2
      }).toList
      builder.deleteCharAt(builder.size-1) //delete last comma in line
      flushLineToDisk(builder)

      listOfItems.transpose.foreach(line => {
        line.foreach(double => {
          builder.append(double.toString.concat(","))
        })
        builder.deleteCharAt(builder.size-1) //delete last comma in line
        flushLineToDisk(builder)
      })

      //      pw.print(builder) //Or b. write at the end the whole thing
      pw.close()
    }

//    interEffp.toList.zipWithIndex.foreach { case (e, i) => {
//      e.toList.zipWithIndex.foreach{case (ee, j) => {
//        println(i, j)
//        traceThinned.predict(interEffp)
//      } }
//    }
//    }

//    val postInters = predictAllInters(traceThinned, interEffp)


    println("ddd")
    //println(postAlphasN)
    val added = List(traceThinned.predict(mu), traceThinned.predict(tauDRV))
//    printSamplesToCsv("/home/antonia/ResultsFromCloud/CompareRainier/040619/withoutInteractions/1M/skataNew.csv", postInters)

    println("mu, taus")
    println(mean(postmuTaus(::, *)))
    println("alphas")
    //println(postAlphasN.map(el=>el._2.sum/el._2.size))
    println("dokimi")
    println(dokimi.map{case(a,b) => (a, b.sum/b.size )})



  }
}

