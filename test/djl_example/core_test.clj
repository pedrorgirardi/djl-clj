(ns djl-example.core-test
  (:require [clojure.test :refer :all]
            [djl-example.core :as djl-example])
  (:import (ai.djl.mxnet.zoo.nlp.qa QAInput)
           (ai.djl Application$NLP)))

(deftest make-criteria-test
  (let [criteria (djl-example/make-criteria {:application Application$NLP/QUESTION_ANSWER
                                             :input QAInput
                                             :output String
                                             :filter {"backbone" "bert"
                                                      "dataset" "book_corpus_wiki_en_uncased"}})]

    (testing "Application"
      (is (= Application$NLP/QUESTION_ANSWER (.getApplication criteria))))

    (testing "Types"
      (is (= QAInput (.getInputClass criteria)))
      (is (= String (.getOutputClass criteria))))

    (testing "Filter"
      (is (= {"backbone" "bert"
              "dataset" "book_corpus_wiki_en_uncased"}
             (.getFilters criteria))))))