import cats._
import cats.implicits._
import com.stripe.rainier.core._
import com.stripe.rainier.compute._
import com.stripe.rainier.sampler._
import com.stripe.rainier.notebook._
import com.cibo.evilplot._
import com.cibo.evilplot.plot._

object NewVRainierEx {

  def anova: Unit = {
    println("anova")
    // simulate synthetic data
    implicit val rng = ScalaRNG(3)
    //val n = 50 // groups
    //val N = 150 // obs per group
    val n = 10 // groups
    val N = 20 // obs per group
    val mu = 5.0 // overall mean
    val sigE = 2.0 // random effect SD
    val sigD = 3.0 // obs SD
    val effects = Vector.fill(n)(sigE * rng.standardNormal)
    val data = effects map (e =>
      Vector.fill(N)(mu + e + sigD * rng.standardNormal))
    println(data)
    // build model
    val m = Normal(0, 100).latent
    val sD = LogNormal(0, 10).latent
    val sE = LogNormal(1, 5).latent
    val eff = Vector.fill(n)(Normal(m, sE).latent)
    val models = (0 until n).map(i =>
      Model.observe(data(i), Normal(eff(i), sD)))
    println(models.size)
    val model = models.reduce{(m1, m2) => m1.merge(m2)}
    println(model.targetGroup.outputs)
    // now sample the model
    //val sampler = EHMC(warmupIterations = 10, iterations = 10)
    println("sampling...")
    val thin = 100
    //HMC(50, 100, 50) each chain 100 samples, 4 chains by default, 50  warmup iterations
    val trace = model.sample(HMC(1000, 100, 50))
    println(trace.chains)
    println("finished sampling.")
    val mt = trace.predict(m)
    show("mu", density(mt))
    //displayPlot(density(mt).render())
  }


  def main(args: Array[String]): Unit = {
    println("main starting")

    //tutorial
    //logReg
    anova


    println("main finishing")
  }
}
