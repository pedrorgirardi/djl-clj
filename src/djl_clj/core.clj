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
           (ai.djl Model Device)
           (ai.djl.basicmodelzoo.basic Mlp)
           (ai.djl.nn Block)
           (ai.djl.training.loss Loss)
           (ai.djl.training TrainingConfig DefaultTrainingConfig Trainer)
           (ai.djl.training.evaluator Evaluator Accuracy)
           (ai.djl.training.listener TrainingListener)
           (ai.djl.metric Metrics)
           (ai.djl.training.dataset Dataset)))

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

(defn devices
  "Returns an array of devices given the maximum number of GPUs to use.

   If GPUs are available, it will return an array
   of Device of size (min num-available max-gpus).
   Else, it will return an array with a single CPU device."
  [max-gpus]
  (Device/getDevices max-gpus))

(defn ^Loss softmax-cross-entropy-loss
  "Returns a new instance of SoftmaxCrossEntropyLoss with default arguments."
  []
  (Loss/softmaxCrossEntropyLoss))

(defn ^Accuracy accuracy-evaluator
  "Returns a multiclass accuracy evaluator that computes accuracy across axis
   1 along the 0th index."
  []
  (Accuracy.))

(defn ^TrainingConfig default-trainning-config
  "A Trainer requires an Initializer to initialize the parameters of the
   model, an Optimizer to compute gradients and update the parameters according
   to a Loss function. It also needs to know the Evaluators that need to be
   computed during training. A TrainingConfig instance that is passed to the
   Trainer will provide this information, and thus facilitate the training
   process."
  [{:keys [loss evaluators devices listeners]}]
  (let [config (DefaultTrainingConfig. loss)]
    (doseq [^Evaluator evaluator evaluators]
      (.addEvaluator config evaluator))

    (when (seq devices)
      (.optDevices config devices))

    (when (seq listeners)
      (.addTrainingListeners config (into-array TrainingListener listeners)))

    config))

(defn ^Metrics metrics
  "A collection of Metric objects organized by metric name.

   Metric is a utility class that is used in the Trainer and Predictor to capture
   performance and other metrics during runtime.

   It is built as a collection of individual Metric classes. As a container
   for individual metrics classes, Metrics stores them as time series data so
   that metric-vs-timeline analysis can be performed. It also provides
   convenient statistical methods for getting aggregated information, such as
   mean and percentile. The metrics is used to store key performance indicators
   (KPIs) during inference and training runs. These KPIs include various
   latencies, CPU and GPU memory consumption, losses, etc."
  []
  (Metrics.))

(defn batches
  "Fetches an iterator that can iterate through the given Dataset.

   Returns an Iterable of Batch that contains batches of data from the dataset."
  [^Trainer trainer ^Dataset dataset]
  (.iterateDataset trainer dataset))

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
