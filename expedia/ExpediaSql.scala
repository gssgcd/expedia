package com.ch.ml.expedia

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

/**
  * Created by ch on 2016/8/31.
  * 使用spark SQL 进行排序
  * expedia 预测结果数据排序，按label分组，按预测值高低排序
  */
object ExpediaSql {

  def main(args: Array[String]) {
    if (args.length != 2) {
      println("Usage:ExpediaTrain <trainPath> <outPath>")
      System.exit(0)
    }
    val Array(trainPath, outPath) = args
    //    val trainPath="I:\\chinahadoop\\机器学习训练营\\训练营作业&代码\\3，推荐系统项目\\xxdata\\output\\part-00000"
    val conf = new SparkConf()
    /*.setAppName("sqlTest").setMaster("local[2]")*/
    val sc = new SparkContext(conf)
    val reRDD = sc.textFile(trainPath) //part-00000.log
    val schemaString = "prob label search hotel"
    val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)
    val rowRDD = reRDD.map(_.replace("(", "").replace(")", "").split(",")).map(attributes => Row(attributes(0), attributes(1), attributes(2), attributes(3)))
    val sqlContext = new SQLContext(sc)

    val rmDF = sqlContext.createDataFrame(rowRDD, schema).repartition(1)
    rmDF.registerTempTable("expdeia")
    rmDF.groupBy()

    val results = sqlContext.sql("SELECT prob,label,search,hotel FROM expdeia order By search asc,prob desc")
    results.map {
      v =>
        v.get(2) + "," + v.get(3)
    }.coalesce(1)
      .saveAsTextFile(outPath)
  }
}
