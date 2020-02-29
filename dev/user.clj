(ns user
  (:require [clojure.datafy :as datafy]
            [clojure.tools.namespace.repl :refer [refresh]]
            [clojure.java.io :as io]

            [djl-clj.core :refer :all])
  (:import (javax.imageio ImageIO)
           (java.io File)

           (ai.djl.mxnet.zoo.nlp.qa QAInput)
           (ai.djl.modality.cv ImageVisualization)
           (ai.djl Application$NLP)))

(comment

  (refresh)

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