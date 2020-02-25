import java.io.File

import breeze.linalg._
import com.cibo.evilplot.numeric.Point
import com.cibo.evilplot.plot.{LinePlot, _}
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._

import scala.annotation.tailrec

object MapAnova2WithoutInters_vComp {

  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val (data, n1, n2) = dataProcessing()
    mainEffects(data, rng, n1, n2)
  }

  /**
   * Process data read from input file
   */
  def dataProcessing(): (Map[(Int, Int), List[Double]], Int, Int) = {
    val data = csvread(new File("/home/antonia/ResultsFromCloud/CompareRainier/040619/withoutInteractions/1M/simulNoInter040619.csv"))
    val sampleSize = data.rows
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
    //println(dataMap)
    (dataMap, nj, nk)

  }

    /**
   * Use Rainier for modelling the main effects only, without interactions
   */
  def mainEffects(dataMap: Map[(Int, Int), List[Double]], rngS: ScalaRNG, n1: Int, n2: Int): Unit = {

    // Implementation of sqrt for Real
    def sqrtR(x: Real): Real = {
      val lx = (Real(0.5) * x.log).exp
      lx
    }
    val varNames = (1 to 29 toList).map(i => i.toString)

    implicit val rng = rngS
    val n = dataMap.size //No of groups
    // All prior values for the unknown parameters, defined as follows, are stored in lists, to be able to process and print the results at the end.
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

    val eff1 = Vector.fill(n1)(Normal(0, sdE1).latent)
    val eff2 = Vector.fill(n2)(Normal(0, sdE2).latent)

    val dataMapKeysToIndexseq = dataMap.keys.toIndexedSeq

    val models = (0 until n).map( z => Model.observe(dataMap(dataMapKeysToIndexseq(z)._1, dataMapKeysToIndexseq(z)._2), Normal(mu + eff1(dataMapKeysToIndexseq(z)._1) + eff2(dataMapKeysToIndexseq(z)._2), sdDR)))

    println(models)
    val model = models.reduce { (m1, m2) => m1.merge(m2) }

    println("sampling...")
    val thin = 100
    //HMC(150, 100, 50) each chain 100 samples, 4 chains by default, 150  warmup iterations, 50 leapfrog steps
    val trace = model.sample(HMC(50, 10, 5))
    val resWithNames = trace.chains.map(l => l.map(ar => varNames.map(name => (name, ar)).toMap))
    println(resWithNames)
  }
}
