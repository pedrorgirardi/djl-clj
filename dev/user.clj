(ns user
  (:require [clojure.datafy :as datafy]
            [clojure.tools.logging :as log]
            [clojure.tools.namespace.repl :refer [refresh]]
            [clojure.java.io :as io]

            [djl-clj.core :refer :all])
  (:import (javax.imageio ImageIO)
           (java.io File)

           (ai.djl.mxnet.zoo.nlp.qa QAInput)
           (ai.djl.modality.cv ImageVisualization)
           (ai.djl Application$NLP Model)
           (ai.djl.basicdataset Mnist)
           (ai.djl.ndarray.types Shape)
           (ai.djl.training.dataset Dataset$Usage Batch)
           (ai.djl.training Trainer)
           (ai.djl.training.listener TrainingListener$Defaults)))

(comment

  (refresh)

  (def block
    (mlp (* Mnist/IMAGE_HEIGHT Mnist/IMAGE_WIDTH) Mnist/NUM_CLASSES [128 64]))

  (def epochs 2)
  (def batch-size 64)

  (def ^Mnist training-set
    (doto (.build (doto (Mnist/builder)
                    (.optUsage (Dataset$Usage/TRAIN))
                    (.setSampling batch-size true)))
      (.prepare (progress-bar))))

  (def ^Mnist test-set
    (doto (.build (doto (Mnist/builder)
                    (.optUsage (Dataset$Usage/TEST))
                    (.setSampling batch-size true)))
      (.prepare (progress-bar))))

  (def config
    (default-trainning-config {:loss (softmax-cross-entropy-loss)
                               :evaluators [(accuracy-evaluator)]
                               :listeners (TrainingListener$Defaults/logging "Mnist training"
                                                                             batch-size
                                                                             (.getNumIterations training-set)
                                                                             (.getNumIterations test-set)
                                                                             nil)}))

  (with-open [^Model model (doto (Model/newInstance) (.setBlock block))
              ^Trainer trainer (doto (.newTrainer model config)
                                 ;; MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
                                 ;; 1st axis is batch axis, we can use 1 for initialization.
                                 (.initialize (into-array Shape [(Shape. [1 (* Mnist/IMAGE_HEIGHT Mnist/IMAGE_WIDTH)])])))]
    (doseq [epoch (range epochs)]
      (doseq [^Batch batch (batch-iterable trainer training-set)]
        (.trainBatch trainer batch)
        (.step trainer)
        (.close batch))

      ;; Reset training and validation evaluators at end of epoch
      (.endEpoch trainer)))


  ;; -- SSD

  (with-open [model (load-model :ssd)
              predictor (predictor model)]
    (let [image (image-from-url (io/resource "soccer.png"))
          detections (predict predictor image)]
      (ImageVisualization/drawBoundingBoxes image detections)
      (ImageIO/write image "png" (File. "ssd.png"))))

  ;; -- Bert QA
  ;; https://github.com/awslabs/djl/blob/master/examples/docs/BERT_question_and_answer.md

  (with-open [model (load-model {:application Application$NLP/QUESTION_ANSWER
                                 :input QAInput
                                 :output String
                                 :progress (progress-bar)
                                 :filter {"backbone" "bert"
                                          "dataset" "book_corpus_wiki_en_uncased"}})
              predictor (predictor model)]
    (let [input (QAInput.
                  "When did BBC Japan start broadcasting?"
                  "BBC Japan was a general entertainment Channel.
                   Which operated between December 2004 and April 2006.
                   It ceased operations after its Japanese distributor folded."
                  384)]
      (predict predictor input)))

  )