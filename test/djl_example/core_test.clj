(ns djl-example.core-test
  (:require [clojure.test :refer :all]
            [djl-example.core :as djl-example])
  (:import (ai.djl.mxnet.zoo.nlp.qa QAInput)
           (ai.djl Application$NLP)
           (clojure.lang ExceptionInfo)))

(deftest build-criteria-test
  (testing "Criteria map"
    (let [criteria (djl-example/build-criteria {:application Application$NLP/QUESTION_ANSWER
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
               (.getFilters criteria)))))))

(deftest load-model-test
  (testing "Try to load using an invalid criteria"
    (is (= (var-get #'djl-example/invalid-criteria-message)
           (try
             (djl-example/load-model nil)
             (catch ExceptionInfo e
               (ex-message e)))))))