(ns user
  (:refer-clojure :exclude [load])
  (:require [clojure.core.protocols :as p]
            [clojure.datafy :as datafy]
            [clojure.tools.namespace.repl :refer [refresh]])
  (:import (javax.imageio ImageIO)
           (java.io File)
           (java.awt.image BufferedImage)

           (ai.djl.modality.cv.util BufferedImageUtils)
           (ai.djl.mxnet.zoo MxModelZoo)
           (ai.djl.training.util ProgressBar)
           (ai.djl.modality.cv ImageVisualization DetectedObjects$DetectedObject)
           (ai.djl.repository.zoo ModelLoader ZooModel)
           (ai.djl.translate Translator)
           (ai.djl.inference Predictor)
           (ai.djl.modality Classifications$Classification Classifications)))

(set! *warn-on-reflection* true)

(defn ^BufferedImage image-from-url [^String url]
  (BufferedImageUtils/fromUrl url))

(defn ^ModelLoader loader [k]
  (case k
    :mx-model-zoo/sdd MxModelZoo/SSD))

(defn ^ZooModel load [^ModelLoader loader]
  (.loadModel loader (ProgressBar.)))

(defn ^ZooModel model [k]
  (load (loader k)))

(defn predictor
  ([^ZooModel model]
   (.newPredictor model))
  ([^ZooModel model ^Translator translator]
   (.newPredictor model translator)))

(defn predict [^Predictor predictor input]
  (.predict predictor input))

(extend-protocol p/Datafiable
  Classifications
  (datafy [o]
    #:djl.classifications{:best (datafy/datafy (.best o))
                          :items (map datafy/datafy (.items o))})

  Classifications$Classification
  (datafy [o]
    #:djl.classification{:class (.getClassName o)
                         :probability (.getProbability o)}))

(comment

  (refresh)

  (def image
    (image-from-url "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/pose/soccer.png"))

  (def model
    (model :mx-model-zoo/sdd))

  (def detections
    (-> (predictor model)
        (predict image)))

  (datafy/datafy detections)


  (ImageVisualization/drawBoundingBoxes image detections)

  (ImageIO/write image "png" (File. "ssd.png"))

  )