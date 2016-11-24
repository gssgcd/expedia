package com.ch.ml.expedia

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ch on 2016/8/25.
  * expedia 模型训练
  */
object ExpediaTrain {

  def main(args: Array[String]) {
    // 在终端商过滤无关日志
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length != 5) {
      println("Usage:ExpediaTrain <trainPath> <testPath> <trainOutput> <modelPath> <flag>")
            System.exit(0)
    }
    val args1 = Array("file:///home/hive/bj/ch/expedia/expedia_ml/part-00000"
      , "file:///home/hive/bj/ch/expedia/expedia_ml_test/part-00000"
      , "file:///home/hive/bj/ch/expedia/expedia_out"
      , "file:///home/hive/bj/ch/expedia/expedia_out_model"
      , "1"
    )
val args2 = Array("i:\\chinahadoop\\机器学习训练营\\训练营作业&代码\\3，推荐系统项目\\xxdata\\expedia_ml\\part-00000"
  , "i:\\chinahadoop\\机器学习训练营\\训练营作业&代码\\3，推荐系统项目\\xxdata\\expedia_ml_test\\part-00000"
  , "i:\\chinahadoop\\机器学习训练营\\训练营作业&代码\\3，推荐系统项目\\expedia_train_out"
  ,"1"
)

    val args3 = Array("/user/s-56/expedia_ml/part-00000"
      , "/user/s-56/expedia_ml_test/part-00000"
      , "/user/s-13/expedia_train_out2"
      , "/user/s-13/expedia_train_out_model2"
      ,"1"
    )
    val Array(trainPath, testPath, trainOutput,modelPath, flag) = args

    val conf = new SparkConf()/* .setAppName("expediaTrain").setMaster("local[2]")*/
    val sc = new SparkContext(conf)

    val trainData = MLUtils.loadLibSVMFile(sc,trainPath)
    val testData = MLUtils.loadLibSVMFile(sc, testPath)
    // train数据3 7 分
    //    val splits = data.randomSplit(Array(0.7, 0.3))
    //    val (trainingData, testData) = (splits(0), splits(1))

    // 把LablePoint数据转换为分层抽样支持的K-V数据
    val preData = trainData.map(x => (x.label, x.features))
    // 对数据进行分层抽样  每一层都抽取0.8
//    val fractions = preData.map(_._1).distinct.map(x => (x, 0.8)).collectAsMap
    val fractions = preData.map(_._1).distinct.map(x => (x, 0.8)).collectAsMap
    // 改造一下，正样本多留点，负样本少取点
    /*val fractions = preData.map(_._1).distinct.map{
      label=>
         label match {
             // 这里必须要加D,表示类型为double，否则，默认是Any类型，编译时报错
           case 0 => (label,0.7D)
           case 1 => (label,1D)
           case _ => (label,0.7D)
        }
    }.collectAsMap*/
    // withReplacement = false 表示不重复抽样
    val sampleData = preData.sampleByKey(withReplacement = false, fractions, seed = 111L).map(x => LabeledPoint(x._1, x._2)).cache()
    //    val fractions: Map[Int, Double] = (List((1, 0.2), (2, 0.8))).toMap //表示在层1抽0.2，在层2中抽0.8
    if (flag.equals("1")) {
      // lr1
      // 训练 LR 模型
      val lrModel = new LogisticRegressionWithLBFGS()
//        .setNumClasses(20)
        .run(sampleData)
      lrModel.save(sc,modelPath)
      // clear the default threshold
      lrModel.clearThreshold()

      //计算泛化误差
      val testErr = testData.map {
        point =>
          val score = lrModel.predict(point.features)
          // prob label search hotel
          score.toFloat + "," + point.label.toInt + "," + point.features(1).toInt + "," + point.features(7).toInt
      }.coalesce(1, true).saveAsTextFile(trainOutput)

      // 把模型保存下来之后，可以进行加载使用
//      lrModel.save(sc, "LR1-model" + outputPath)


      // 可以读
//      val savedLRModel = LogisticRegressionModel.load(sc,"lrModel_path")
    } else if (flag.equals("2")) {
      //lr2
      val lrModel2 = new LogisticRegressionWithSGD().run(sampleData)
      lrModel2.clearThreshold()

      // LR model 2
      testData.map {
        point =>
          val score = lrModel2.predict(point.features) //prob label search hotel
          score.toFloat + "," + point.label.toInt + "," + point.features(1).toInt + "," + point.features(7).toInt
      }.coalesce(1, true).saveAsTextFile(trainOutput)

//      lrModel2.save(sc,  modelPath)

    } else if (flag.equals("3")) {
      // gbdt
      // 训练GBDT模型
      val boostingStrategy = BoostingStrategy.defaultParams("Classification") /*.setNumIterations(3)*/
      val GBDTModel = GradientBoostedTrees.train(sampleData, boostingStrategy)

      // GBDTModel
      testData.map {
        point =>
          val score = GBDTModel.predict(point.features)
          score.toFloat + "," + point.label.toInt + "," + point.features(1).toInt + "," + point.features(7).toInt
      }.coalesce(1, true).saveAsTextFile(trainOutput)

//      GBDTModel.save(sc, modelPath)
    }





    // 计算测试误差
    /*   val testErr = testData.map { point => val prediction = model.predict(point.features)
         if (point.label == prediction) 1.0 else 0.0
       }.mean()
       println("Test Error = " + testErr)
       println("Learned GBT model:n" + model.toDebugString)*/

    // 0.9720347763749159
  }
}
