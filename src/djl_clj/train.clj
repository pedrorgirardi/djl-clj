(ns djl-clj.train
  (:require [clojure.tools.logging :as log]
            [djl-clj.core :as djl])
  (:import (java.nio.file Paths)
           (ai.djl.training.dataset Batch Dataset$Usage)
           (ai.djl.basicdataset Mnist)
           (ai.djl Device Model)
           (ai.djl.training.listener TrainingListener$Defaults)
           (ai.djl.training Trainer)
           (ai.djl.ndarray.types Shape)))

(def epochs 2)

(def batch-size 32)

(defn ^Mnist training-set []
  (doto (.build (doto (Mnist/builder)
                  (.optUsage (Dataset$Usage/TRAIN))
                  (.setSampling batch-size true)))
    (.prepare (djl/progress-bar))))

(defn ^Mnist test-set []
  (doto (.build (doto (Mnist/builder)
                  (.optUsage (Dataset$Usage/TEST))
                  (.setSampling batch-size true)))
    (.prepare (djl/progress-bar))))

(defn block []
  (djl/mlp (* Mnist/IMAGE_HEIGHT Mnist/IMAGE_WIDTH) Mnist/NUM_CLASSES [128 64]))

(def config
  (djl/default-trainning-config (djl/softmax-cross-entropy-loss)
                                {:evaluators [(djl/accuracy-evaluator)]
                                 :devices [(Device/cpu)]
                                 :listeners (TrainingListener$Defaults/logging)}))

(defn -main []
  (with-open [^Model model (doto (Model/newInstance)
                             (.setBlock (block)))

              ^Trainer trainer (doto (.newTrainer model config)
                                 (.setMetrics (djl/metrics))
                                 (.initialize (into-array Shape [(Shape. [1 (* Mnist/IMAGE_HEIGHT Mnist/IMAGE_WIDTH)])])))]

    (doseq [_ (range epochs)]
      (doseq [^Batch batch (djl/batches trainer (training-set))]
        (try
          (.trainBatch trainer batch)
          (.step trainer)
          (catch Exception e
            (log/error e))
          (finally
            (.close batch))))

      (doseq [^Batch batch (djl/batches trainer (test-set))]
        (try
          (.validateBatch trainer batch)
          (catch Exception e
            (log/error e))
          (finally
            (.close batch))))

      ;; Reset training and validation evaluators at end of epoch
      (.endEpoch trainer))

    (.setProperty model "Epoch" (str epochs))
    (.save model (Paths/get "temp" (into-array [""])) "mnist"))

  (System/exit 0))
