(ns djl-clj.core
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
           (ai.djl.util Progress)
           (ai.djl Model)
           (ai.djl.basicmodelzoo.basic Mlp)
           (ai.djl.nn Block)))

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

(defn ^Block mlp
  "A Multilayer Perceptron (MLP) NeuralNetwork using RELU.

   => (mlp 784 10 [128 64])"
  [input-size output-size hidden-sizes]
  (Mlp. input-size output-size (into-array Integer/TYPE hidden-sizes)))

(defn ^Criteria build-criteria [{:keys [application input output progress filter]}]
  (let [^Criteria$Builder builder (.setTypes (Criteria/builder) input output)]
    (when application
      (.optApplication builder application))

    (when progress
      (.optProgress builder progress))

    (when filter
      (doseq [[k v] filter]
        (.optFilter builder k v)))

    (.build builder)))

(defn ^Progress progress-bar
  "ProgressBar is an implementation of Progress.
   It can be used to display the progress of a task in the form a bar."
  []
  (ProgressBar.))

(def loaders
  {:ssd MxModelZoo/SSD
   :bert-qa MxModelZoo/BERT_QA})

(def available-loaders
  (keys loaders))

(defn ^ModelLoader model-loader
  "A ModelLoader loads a particular ZooModel from a Repository for a model zoo.

   See `available-loaders` for available keywords."
  [k]
  (or (loaders k) (throw (ex-info (str "Loader " k " doesn't exist.") {:available-loaders available-loaders}))))

(def ^:private invalid-criteria-message
  "Invalid criteria. It must be either a keyword, map or ai.djl.repository.zoo.Criteria.
   See `available-loaders` for available keywords.")

(defn ^ZooModel load-model
  "Returns the model that matches criteria.

   Where `criteria` is:
   - a keyword (see `available-loaders`)
   - or a map;
   - or a ai.djl.repository.zoo.Criteria"
  [criteria]
  (cond
    (keyword? criteria)
    (.loadModel (model-loader criteria) (progress-bar))

    (map? criteria)
    (ModelZoo/loadModel (build-criteria criteria))

    (instance? Criteria criteria)
    (ModelZoo/loadModel criteria)

    :else
    (throw (ex-info invalid-criteria-message {}))))

(defn predictor
  "Creates a new Predictor based on the model.

   If it's a `ai.djl.repository.zoo.ZooModel` a default translator will be used.

   You can use a Predictor, with a specified Translator, to perform inference on a Model."
  ([^ZooModel model]
   (.newPredictor model))
  ([^Model model ^Translator translator]
   (.newPredictor model translator)))

(defn predict
  "Predicts an item for inference."
  [^Predictor predictor input]
  (.predict predictor input))
