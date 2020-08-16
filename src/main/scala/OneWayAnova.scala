import com.stripe.rainier.core._
import com.stripe.rainier.sampler._

/**
 * Builds a simple 1-way Anova model in Scala using Rainier version 0.3.2
 */
object OneWayAnova {
  def anova: Unit = {
    println("anova")

    // Simulate synthetic data
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

    // Define parameters and priors
    val mu = Normal(0, 100).latent
    val sdObs = LogNormal(0, 10).latent
    val sdEff = LogNormal(1, 5).latent
    val eff = Vector.fill(n)(Normal(mu, sdEff).latent)

    // Update by observing the data per group
    val models = (0 until n).map(i =>
      Model.observe(data(i), Normal(eff(i), sdObs)))

    // Merge the different models
    val model = models.reduce{(m1, m2) => m1.merge(m2)}

    // Fit the model
    val iterations = 2500
    val lfrogStep = 50

    // The new implementantion of HMC runs 4 parallel chains
    val output = model.sample(HMC(warmupIterations = 1000, iterations, lfrogStep))
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
    time(anova)
    println("main finishing")
  }
}
