(ns user
  (:require [clojure.datafy :as datafy]
            [clojure.tools.namespace.repl :refer [refresh]]
            [clojure.java.io :as io]

            [djl-example.core :refer :all])
  (:import (javax.imageio ImageIO)
           (java.io File)

           (ai.djl.modality.cv ImageVisualization)))

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

  )