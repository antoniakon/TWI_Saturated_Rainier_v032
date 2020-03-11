import java.io.{BufferedWriter, File, FileWriter, PrintWriter}

import breeze.linalg._
import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._

import scala.collection.immutable.ListMap

object MapAnova2WithoutInters_vComp {

  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val (data, dataRaw, n1, n2) = dataProcessing()
    mainEffects(data, dataRaw, rng, n1, n2)
  }

  /**
   * Process data read from input file
   */
  def dataProcessing(): (Map[(Int, Int), List[Double]], (Array[Double], Array[Int], Array[Int]), Int, Int) = {
    val data = csvread(new File("./SimulatedDataAndTrueCoefs/simulDataNoInters.csv"))
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

    val dataMap = (dataList zip y).groupBy(_._1).map { case (k, v) => ((k._1 - 1, k._2 - 1), v.map(_._2)) } //Bring the data to the map format
    val dataRaw = (y.map(i => i-1), alpha.toArray.map(i => i-1), beta.toArray.map(i => i-1))
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

    //Create a Vec[Real] for each effect
    val eff1p = Normal(0, sdE1).latentVec(n1)
    val eff2p = Normal(0, sdE2).latentVec(n2)

    /**
     * Uses the imported data in format (y, alpha, beta)
     * 1. Creates an indexedSeq[Real] with the group means.
     * 2. Turns this to Vec
     * 3. maps over the vec to create a normal for each observation
     * 4. Builds the model
     *
     * PROBLEM: Does not do anything
     */
    def implementation1(): Model ={
      val allEffs = (0 until obs.size).map(i => mu + eff1p(alpha(i)) + eff2p(beta(i))) //1.
      val allEffsVec = Vec.from(allEffs) //2.
      val modelVec = allEffsVec.map{i:Real => Normal(i, sdDR)} //3.
      val vecModel = Model.observe(obs.toList, modelVec) //4.
      vecModel
    }

    /**
     * Uses the imported data in format (y, alpha, beta)
     * 1. Creates a Vec[Real] for columns alpha & beta
     * 2. zips the vecs to one Vec[(Real, Real)] to map over
     * 3. maps over the Vec[(Real, Real)] to create a normal for each observation
     * 4. Builds the model
     *
     * Problem: Slow
     * e.g. For HMC(100, 100, 50) time: 380505.616669ms
     * For HMC(1000, 1000, 100) 6788099.496656ms
     */
    def implementation2(): Model = {
      val eff1Vec = Vec.from(alpha) //1.
      val eff2Vec = Vec.from(beta) //1.
      val eff1VecZipEff2Vec = eff1Vec zip eff2Vec //2.
      val modeleff1Vec = eff1VecZipEff2Vec.map{case (a,b) => Normal(mu + eff1p(a) + eff2p(b), sdDR)} //3.
      val vecModel = Model.observe(obs.toList, modeleff1Vec) //4.
      vecModel
    }

    /**
     * Uses the imported data in format Map[(Int, Int), List[Double]]
     * 1. Creates an indexedSeq[(Int, Int)] for the keys = groups. No of groups n = n1 x n2
     * 2. Creates the model per group
     * 3. Merges all the models
     *
     * PROBLEM: Slow and wrong results?
     */
    def implementation3(): Model = {
      val dataMapKeysToIndexseq = dataMap.keys.toIndexedSeq //1.
      val models = (0 until n).map( z => Model.observe(dataMap(dataMapKeysToIndexseq(z)._1, dataMapKeysToIndexseq(z)._2), Normal(mu + eff1p(dataMapKeysToIndexseq(z)._1) + eff2p(dataMapKeysToIndexseq(z)._2), sdDR))) //2.
      val model = models.reduce { (m1, m2) => m1.merge(m2) } //3.
      model
    }

    // Calculation of the execution time
    def time[A](f: => A): A = {
      val s = System.nanoTime
      val ret = f
      val execTime = (System.nanoTime - s) / 1e6
      println("time: " + execTime + "ms")
      val bw = new BufferedWriter(new FileWriter(new File("./SimulatedDataAndTrueCoefs/results/RainierResWithoutInterHMC200-1mTimeNewVersion.txt")))
      bw.write(execTime.toString)
      bw.close()
      ret
    }

    println("sampling...")
    val vecModel = implementation2()
    val warmupIters = 1000
    val samplesPerChain = 500
    val leapfrogSteps = 100
    val thinBy = 10
    val trace = time(vecModel.sample(HMC(warmupIters, samplesPerChain, leapfrogSteps)))
    println("sampling over...")
    val traceThinned = trace.thin(thinBy)
    val sampNo = samplesPerChain * 4 / thinBy


    def predictMainEffs(myTrace: Trace, vc: Vec[Real], name: String) : Map[String, List[Double]] = {
     vc.toList.zipWithIndex.map{
       case (e, i) => { name.concat(i.toString) -> myTrace.predict(e) }
     }.toMap
    }

    def predictmuSigmas(myTrace: Trace, r: Real, name: String) : Map[String, List[Double]] = {
      Map(name-> myTrace.predict(r))
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

    val postmu = predictmuSigmas(traceThinned, mu, "mu")
    val postsd = predictmuSigmas(traceThinned, sdDR, "sdDR")
    val postsdE1 = predictmuSigmas(traceThinned, sdE1, "sdE1")
    val postsdE2 = predictmuSigmas(traceThinned, sdE2, "sdE2")
    val postAlphas = predictMainEffs(traceThinned, eff1p, "effA")
    val postBetas = predictMainEffs(traceThinned, eff2p, "effB")

    val muSigmas = postmu ++ postsd ++ postsdE1 ++ postsdE2
    val allRes =  muSigmas ++ postAlphas ++ postBetas
    printSamplesToCsv("./SimulatedDataAndTrueCoefs/results/RainierResWithoutInterHMC200-1mTimeNewVersion.csv", allRes)

    println("muSigmas")
    println(muSigmas.map(el =>el._2.sum/el._2.size))
    println("alphas")
    println(postAlphas.map(el=>el._2.sum/el._2.size))
    println("betas")
    println(postBetas.map(el=>el._2.sum/el._2.size))
  }
}

