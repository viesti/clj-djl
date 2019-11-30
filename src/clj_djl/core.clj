(ns clj-djl.core
  (:require [clojure.java.io :as io])
  (:import (ai.djl.modality.cv.util BufferedImageUtils)
           (ai.djl.mxnet.zoo MxModelZoo)
           (ai.djl.training.util ProgressBar)
           (ai.djl.modality.cv ImageVisualization)
           (javax.imageio ImageIO)))

(defn example []
  (let [img (BufferedImageUtils/fromUrl "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/pose/soccer.png")]
    (with-open [model (.loadModel (MxModelZoo/SSD) (ProgressBar.))]
      (let [predict-result (-> model
                               (.newPredictor)
                               (.predict img))]
        (ImageVisualization/drawBoundingBoxes img predict-result)
        (ImageIO/write img "png" (io/file "ssd.png"))))))
