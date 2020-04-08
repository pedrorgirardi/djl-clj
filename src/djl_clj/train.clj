(ns djl-clj.train
  (:require [clojure.tools.logging :as log])
  (:import (java.nio.file Paths)
           (ai.djl.training.dataset Batch)
           (ai.djl.basicdataset Mnist Mnist$Builder)
           (ai.djl Device Model)
           (ai.djl.training.listener TrainingListener$Defaults TrainingListener)
           (ai.djl.training Trainer DefaultTrainingConfig)
           (ai.djl.ndarray.types Shape)
           (ai.djl.training.util ProgressBar)
           (ai.djl.basicmodelzoo.basic Mlp)
           (ai.djl.training.evaluator Evaluator Accuracy)
           (ai.djl.training.loss Loss)
           (ai.djl.metric Metrics)))

(def epochs 2)

(def batch-size 32)

(defn ^Mnist mnist []
  (let [^Mnist$Builder builder (doto (Mnist/builder)
                                 (.setSampling batch-size true))

        ^Mnist dataset (.build builder)]
    (doto dataset
      (.prepare (ProgressBar.)))))

(defn block []
  (let [input (* Mnist/IMAGE_HEIGHT Mnist/IMAGE_WIDTH)
        output Mnist/NUM_CLASSES
        hidden (into-array Integer/TYPE [128 64])]
    (Mlp. input output hidden)))

(defn config []
  (doto (DefaultTrainingConfig. (Loss/softmaxCrossEntropyLoss))
    (.addEvaluator (Accuracy.))
    (.addTrainingListeners (into-array TrainingListener (TrainingListener$Defaults/logging)))
    (.optDevices (into-array Device [(Device/cpu)]))))

(defn -main []
  (with-open [^Model model (doto (Model/newInstance)
                             (.setBlock (block)))

              ^Trainer trainer (doto (.newTrainer model (config))
                                 (.setMetrics (Metrics.))
                                 (.initialize (into-array Shape [(Shape. [1 (* Mnist/IMAGE_HEIGHT Mnist/IMAGE_WIDTH)])])))]

    (doseq [_ (range epochs)]
      (run!
        (fn [^Batch batch]
          (try
            (.trainBatch trainer batch)
            (.step trainer)
            (catch Exception e
              (log/error e))
            (finally
              (.close batch))))
        (.iterateDataset trainer (mnist)))

      ;; Reset training and validation evaluators at end of epoch
      (.endEpoch trainer))

    (.setProperty model "Epoch" (str epochs))
    (.save model (Paths/get "temp" (into-array [""])) "mnist"))

  (System/exit 0))
