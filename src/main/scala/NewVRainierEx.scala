import com.stripe.rainier.core._
import com.stripe.rainier.sampler._

object NewVRainierEx {

  /**
   * D. Wilkinson's example
   */
  def anova: Unit = {
    println("anova")

    // simulate synthetic data
    implicit val rng = ScalaRNG(3)
    val n = 50 // groups
    val N = 250 // obs per group
    val mu1 = 5.0 // overall mean
    val sigE = 2.0 // random effect SD
    val sigD = 3.0 // obs SD
    val effects = Vector.fill(n)(sigE * rng.standardNormal)
    val data = effects map (e =>
      Vector.fill(N)(mu1 + e + sigD * rng.standardNormal))
    println(data)

    // build the model
    val mu = Normal(0, 100).latent
    val sdObs = LogNormal(0, 10).latent
    val sdEff = LogNormal(1, 5).latent
    val eff = Vector.fill(n)(Normal(mu, sdEff).latent)
    val models = (0 until n).map(i =>
      Model.observe(data(i), Normal(eff(i), sdObs)))
    // println(models.size)
    val model = models.reduce{(m1, m2) => m1.merge(m2)}
    println(model.targetGroup.outputs)

    // now sample the model
    val sampler = EHMC(warmupIterations = 10, iterations = 10)
    println("sampling...")
    val thin = 100
    //HMC(warmupIterations: Int, iterations: Int, nSteps: Int)
    //HMC(50, 100, 50) each chain 100 samples, 4 chains by default, 50  warmup iterations
    val trace = time(model.sample(HMC(10000, 250000, 50)))
    println(trace.chains)
    println("finished sampling.")
//    val mt = trace.predict(mu)
//    show("mu", density(mt))
//    displayPlot(density(mt).render())
  }

  // Calculation of the execution time
  def time[A](f: => A): A = {
    val s = System.nanoTime
    val ret = f
    val execTime = (System.nanoTime - s) / 1e6
    println("time: " + execTime + "ms")
    ret
  }

  def main(args: Array[String]): Unit = {
    println("main starting")
    anova
    println("main finishing")
  }
}
