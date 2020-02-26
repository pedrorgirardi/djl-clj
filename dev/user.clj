(ns user
  (:require [clojure.datafy :as datafy]
            [clojure.tools.namespace.repl :refer [refresh]]
            [clojure.java.io :as io]

            [djl-example.core :refer :all])
  (:import (javax.imageio ImageIO)
           (java.io File)

           (ai.djl.mxnet.zoo.nlp.qa QAInput)
           (ai.djl.modality.cv ImageVisualization)
           (ai.djl Application$NLP)))

(comment

  (refresh)

  (def image
    (image-from-url (io/resource "soccer.png")))

  (def model
    (load-model :ssd))

  (def detections
    (-> (predictor model)
        (predict image)))

  (datafy/datafy detections)

  (ImageVisualization/drawBoundingBoxes image detections)

  (ImageIO/write image "png" (File. "ssd.png"))


  ;; -- Bert QA
  ;; https://github.com/awslabs/djl/blob/master/examples/docs/BERT_question_and_answer.md

  (def model
    (load-model :bert-qa))

  (def model
    (load-model {:application Application$NLP/QUESTION_ANSWER
                 :input QAInput
                 :output String
                 :progress (progress-bar)
                 :filter {"backbone" "bert"
                          "dataset" "book_corpus_wiki_en_uncased"}}))

  (def input
    (QAInput.
      "When did BBC Japan start broadcasting?"
      "BBC Japan was a general entertainment Channel.
       Which operated between December 2004 and April 2006.
       It ceased operations after its Japanese distributor folded."
      384))

  (-> (predictor model)
      (predict input)))