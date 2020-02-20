(ns user
  (:import (ai.djl.modality.cv.util BufferedImageUtils)
           (ai.djl.mxnet.zoo MxModelZoo)
           (ai.djl.training.util ProgressBar)
           (ai.djl.modality.cv ImageVisualization)

           (javax.imageio ImageIO)
           (java.io File)))

(set! *warn-on-reflection* true)

(comment

  (def img (BufferedImageUtils/fromUrl "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/pose/soccer.png"))

  (def model (. MxModelZoo/SSD loadModel (ProgressBar.)))

  (def predict (-> (.newPredictor model) (.predict img)))

  (ImageVisualization/drawBoundingBoxes img predict)

  (ImageIO/write img "png" (File. "ssd.png"))

  )