(ns djl-clj.model-test
  (:require [clojure.test :refer :all]
            [djl-clj.core :as djl])
  (:import (ai.djl.modality.nlp.qa QAInput)
           (ai.djl Application$NLP)))

(deftest bert-qa-test
  (testing "Bert QA - Keyword criteria"
    (with-open [model (djl/load-model :bert-qa)
                predictor (djl/predictor model)]
      (let [input (QAInput.
                    "When did BBC Japan start broadcasting?"
                    "BBC Japan was a general entertainment Channel.
                     Which operated between December 2004 and April 2006.
                     It ceased operations after its Japanese distributor folded."
                    384)]
        (is (= "[december, 2004]" (djl/predict predictor input))))))

  (testing "Bert QA - Map criteria"
    (with-open [model (djl/load-model {:application Application$NLP/QUESTION_ANSWER
                                       :input QAInput
                                       :output String
                                       :progress (djl/progress-bar)
                                       :filter {"backbone" "bert"
                                                "dataset" "book_corpus_wiki_en_uncased"}})
                predictor (djl/predictor model)]
      (let [input (QAInput.
                    "When did BBC Japan start broadcasting?"
                    "BBC Japan was a general entertainment Channel.
                     Which operated between December 2004 and April 2006.
                     It ceased operations after its Japanese distributor folded."
                    384)]
        (is (= "[december, 2004]" (djl/predict predictor input)))))))
