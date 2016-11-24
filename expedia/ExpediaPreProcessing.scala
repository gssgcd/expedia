package com.ch.ml.expedia

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.ArrayBuffer

/**
  * Created by ch on 2016/8/25.
  * expedia 酒店预测  数据预处理
  */
object ExpediaPreProcessing {

  def main(args: Array[String]) {
    // 在终端商过滤无关日志
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length != 4) {
      println("Usage:ExpediaTrain <trainPath> <testPath> <processTrainOut> <processTestOut>")
      System.exit(0)
    }

    val args1=Array("file:///I:\\chinahadoop\\机器学习训练营\\训练营作业&代码\\3，推荐系统项目\\train.csv"
    ,"file:///I:\\chinahadoop\\机器学习训练营\\训练营作业&代码\\3，推荐系统项目\\test.csv"
    ,"train"
    ,"test")
    val Array(trainPath, testPath, processTrainOut, processTestOut) = args

    // 设置参数
    val conf = new SparkConf()/*.setAppName("expedia").setMaster("local[2]")*/
    val sc = new SparkContext(conf)

    //修改train数据 / test
    val data_train = sc.textFile(trainPath)
    val data_test = sc.textFile(testPath)

    // 数据预处理 并存文件
    dataPreProcessing(data_train,"train").coalesce(1,true).saveAsTextFile(processTrainOut)
    dataPreProcessing(data_test,"test").coalesce(1,true).saveAsTextFile(processTestOut)

    sc.stop()
  }

  /**
    * 数据预处理 把csv文件格式转换为libsvm 文件格式
    *
    * @param fileType ""
    */
  def dataPreProcessing(rdd: RDD[String], fileType: String) = {
    //    数据处理 去除null值
    val dataRDD = rdd.filter(!_.contains("srch_id")).map {
      _.replace("NULL", "0").split(",")
    }
    //删除日期列:
    val data5date = dataRDD.map {
      x =>
        val x2 = x.toBuffer
        x2.remove(1, 1) // 下标1开始，删除1列
        x2.toArray
    }

    //转换为libsvm格式: label k1:v1 k2:v2
    val dataLibsvm = data5date.map {
      x =>
        val a = new ArrayBuffer[String]

        if(fileType.equals("train")){ // 把是否预定当做label
        a += x.last
        }else{ // test集没有 booking_bool列，虽然libsvm label可以为空，但spark处理 libsvm数据要有label列,所以test集都设置label为1
        a += "1"
        }
        // 正常这里是1，但是train和test数据集的列数不一样，train比test多了后三列，为了predit时正确，就把train数据集后三列删除
        var clnNum = 1
        if (fileType.equals("train")) {
          clnNum = 4
        }
        for (i <- 0 to x.length - clnNum) {
          // 为了防止index出现-1等非法数据，这里对index统一+2
          // libsvm 数据的index貌似不能以0 开始
          a += (i+2) + ":" + x(i)
        }
        a.toArray.mkString(" ")
    }
    dataLibsvm
  }
}
