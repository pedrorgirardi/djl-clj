(ns djl-clj.core-test
  (:require [clojure.test :refer :all]
            [djl-clj.core :as djl])
  (:import (ai.djl.mxnet.zoo.nlp.qa QAInput)
           (ai.djl Application$NLP)
           (clojure.lang ExceptionInfo)))

(deftest build-criteria-test
  (testing "Criteria map"
    (let [criteria (djl/build-criteria {:application Application$NLP/QUESTION_ANSWER
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
    (is (= (var-get #'djl/invalid-criteria-message)
           (try
             (djl/load-model nil)
             (catch ExceptionInfo e
               (ex-message e)))))))

(deftest default-trainning-config-test
  (let [loss (djl/softmax-cross-entropy-loss)

        config (djl/default-trainning-config {:loss loss
                                              :evaluators [(djl/accuracy-evaluator)]
                                              :devices (djl/devices 0)
                                              :listeners []})]
    (is (= loss (.getLossFunction config)))
    (is (= 1 (count (.getEvaluators config))))
    (is (= 1 (count (.getDevices config))))
    (is (= 0 (count (.getTrainingListeners config))))))