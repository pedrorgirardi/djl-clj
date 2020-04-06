(ns user
  (:require [clojure.tools.logging :as log]
            [clojure.tools.namespace.repl :refer [refresh]]
            [clojure.java.io :as io]

            [djl-clj.core :as djl])
  (:import (javax.imageio ImageIO)
           (java.io File)

           (ai.djl.modality.nlp.qa QAInput)
           (ai.djl.modality.cv ImageVisualization)
           (ai.djl Application$NLP Model Device)
           (ai.djl.basicdataset Mnist)
           (ai.djl.ndarray.types Shape)
           (ai.djl.training.dataset Dataset$Usage Batch)
           (ai.djl.training Trainer)
           (ai.djl.training.listener TrainingListener$Defaults)
           (ai.djl.nn SequentialBlock Blocks Activation)
           (ai.djl.nn.core Linear)
           (java.nio.file Paths)))

(comment

  (refresh)

  ;; Create your first deep learning neural network
  ;;
  ;; https://github.com/awslabs/djl/blob/master/jupyter/tutorial/create_your_first_network.ipynb

  ;; -- Determine your input and output size

  (def input-size (* 28 28))
  (def output-size 10)

  ;; -- Create a SequentialBlock

  (def block (SequentialBlock.))

  ;; -- Add blocks to SequentialBlock
  ;;
  ;; The first layer and last layer have fixed sizes depending on your desired input and output size.
  ;; However, you are free to choose the number and sizes of the middle layers in the network.
  ;; We will create a smaller MLP with two middle layers that gradually decrease the size.
  ;; Typically, you would experiment with different values to see what works the best on your data set.

  (doto block
    (.add (Blocks/batchFlattenBlock input-size))
    ;; -->
    (.add (.build (doto (Linear/builder) (.setOutChannels 128))))
    (.add (reify java.util.function.Function
            (apply [this array]
              (Activation/relu array))))
    ;; -->
    (.add (.build (doto (Linear/builder) (.setOutChannels 64))))
    (.add (reify java.util.function.Function
            (apply [this array]
              (Activation/relu array))))
    ;; -->
    (.add (.build (doto (Linear/builder) (.setOutChannels output-size)))))



  ;; Train your first model
  ;;
  ;; https://github.com/awslabs/djl/blob/ed0e90a32c208b4cf2e5331788eab84b660697cd/jupyter/tutorial/train_your_first_model.ipynb

  (def epochs 2)
  (def batch-size 32)

  (def ^Mnist training-set
    (doto (.build (doto (Mnist/builder)
                    (.optUsage (Dataset$Usage/TRAIN))
                    (.setSampling batch-size true)))
      (.prepare (djl/progress-bar))))

  (def ^Mnist test-set
    (doto (.build (doto (Mnist/builder)
                    (.optUsage (Dataset$Usage/TEST))
                    (.setSampling batch-size true)))
      (.prepare (djl/progress-bar))))

  (def block
    (djl/mlp (* Mnist/IMAGE_HEIGHT Mnist/IMAGE_WIDTH) Mnist/NUM_CLASSES [128 64]))

  (def config
    (djl/default-trainning-config {:loss (djl/softmax-cross-entropy-loss)
                                   :evaluators [(djl/accuracy-evaluator)]
                                   :devices [(Device/cpu)]
                                   :listeners (TrainingListener$Defaults/logging)}))

  (with-open [^Model model (doto (Model/newInstance)
                             (.setBlock block))

              ^Trainer trainer (doto (.newTrainer model config)
                                 (.setMetrics (djl/metrics))
                                 (.initialize (into-array Shape [(Shape. [1 (* Mnist/IMAGE_HEIGHT Mnist/IMAGE_WIDTH)])])))]

    (doseq [epoch (range epochs)]
      (doseq [^Batch batch (djl/batches trainer training-set)]
        (try
          (.trainBatch trainer batch)
          (.step trainer)
          (catch Exception e
            (log/error e))
          (finally
            (.close batch))))

      (doseq [^Batch batch (djl/batches trainer test-set)]
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