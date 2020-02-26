(ns djl-example.core
  (:require [clojure.core.protocols :as p]
            [clojure.datafy :as datafy])
  (:import (java.net URL)
           (java.awt.image BufferedImage)

           (ai.djl.modality Classifications$Classification Classifications)
           (ai.djl.modality.cv.util BufferedImageUtils)
           (ai.djl.mxnet.zoo MxModelZoo)
           (ai.djl.training.util ProgressBar)
           (ai.djl.repository.zoo ModelLoader ZooModel Criteria ModelZoo Criteria$Builder)
           (ai.djl.translate Translator)
           (ai.djl.inference Predictor)
           (ai.djl.util Progress)))

(set! *warn-on-reflection* true)

(extend-protocol p/Datafiable
  Classifications
  (datafy [o]
    #:djl.classifications{:best (datafy/datafy (.best o))
                          :items (map datafy/datafy (.items o))})

  Classifications$Classification
  (datafy [o]
    #:djl.classification{:class (.getClassName o)
                         :probability (.getProbability o)}))

(defn ^BufferedImage image-from-url [^URL url]
  (BufferedImageUtils/fromUrl url))

(defn ^Criteria make-criteria [{:keys [application input output progress filter]}]
  (let [^Criteria$Builder builder (.setTypes (Criteria/builder) input output)]
    (when application
      (.optApplication builder application))

    (when progress
      (.optProgress builder progress))

    (when filter
      (doseq [[k v] filter]
        (.optFilter builder k v)))

    (.build builder)))

(defn ^Progress progress-bar []
  (ProgressBar.))

(def loaders
  {:ssd MxModelZoo/SSD
   :bert-qa MxModelZoo/BERT_QA})

(def available-loaders
  (keys loaders))

(defn ^ModelLoader loader [k]
  (or (loaders k) (throw (ex-info (str "Loader " k " doesn't exist.") {:available-loaders available-loaders}))))

(defn ^ZooModel load-model [criteria]
  (cond
    (keyword? criteria)
    (.loadModel (loader criteria) (progress-bar))

    (map? criteria)
    (ModelZoo/loadModel (make-criteria criteria))

    :else
    (throw (ex-info "Invalid criteria. It must be either a keyword or a map." {}))))

(defn predictor
  ([^ZooModel model]
   (.newPredictor model))
  ([^ZooModel model ^Translator translator]
   (.newPredictor model translator)))

(defn predict [^Predictor predictor input]
  (.predict predictor input))
