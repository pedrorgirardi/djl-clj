(ns djl-clj.model-test
  (:require [clojure.test :refer :all]
            [djl-clj.core :as djl])
  (:import (ai.djl.mxnet.zoo.nlp.qa QAInput)
           (ai.djl Application$NLP)))

(deftest bert-qa-test
  (testing "Bert QA - Keyword criteria"
    (let [input (QAInput.
                  "When did BBC Japan start broadcasting?"
                  "BBC Japan was a general entertainment Channel.
                   Which operated between December 2004 and April 2006.
                   It ceased operations after its Japanese distributor folded."
                  384)

          model (djl/load-model :bert-qa)]
      (is (= "[december, 2004]" (-> (djl/predictor model)
                                    (djl/predict input))))))

  (testing "Bert QA - Map criteria"
    (let [input (QAInput.
                  "When did BBC Japan start broadcasting?"
                  "BBC Japan was a general entertainment Channel.
                   Which operated between December 2004 and April 2006.
                   It ceased operations after its Japanese distributor folded."
                  384)

          model (djl/load-model {:application Application$NLP/QUESTION_ANSWER
                                 :input QAInput
                                 :output String
                                 :progress (djl/progress-bar)
                                 :filter {"backbone" "bert"
                                          "dataset" "book_corpus_wiki_en_uncased"}})]
      (is (= "[december, 2004]" (-> (djl/predictor model)
                                    (djl/predict input)))))))