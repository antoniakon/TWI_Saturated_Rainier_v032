name := "RainierNewVersion"

version := "0.1"

scalaVersion := "2.12.10"

resolvers += Resolver.bintrayRepo("cibotech", "public") // for EvilPlot
//Dependencies for the new version of Rainier
libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.8" % "test",
  "org.scalactic" %% "scalactic" % "3.0.8" % "test",
  "org.typelevel" %% "cats-core" % "2.0.0",
  "org.typelevel" %% "discipline-core" % "1.0.0",
  "org.typelevel" %% "discipline-scalatest" % "1.0.0",
  "org.typelevel" %% "simulacrum" % "1.0.0",
  "com.cibo" %% "evilplot" % "0.6.3", // 0.7.0
  "com.cibo" %% "evilplot-repl" % "0.6.3", // 0.7.0
  "com.stripe" %% "rainier-core" % "0.3.0",
  "com.stripe" %% "rainier-notebook" % "0.3.0",
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/",
  "jitpack" at "https://jitpack.io" // for Jupiter/notebook
)
