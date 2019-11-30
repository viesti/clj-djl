(ns clj-djl.core
  (:require [clojure.java.io :as io])
  (:import (ai.djl.modality.cv.util BufferedImageUtils)
           (ai.djl.mxnet.zoo MxModelZoo)
           (ai.djl.training.util ProgressBar)
           (ai.djl.modality.cv ImageVisualization)
           (javax.imageio ImageIO)
           (java.awt.image BufferedImage)))

(defn detect-image-objects [^BufferedImage source dest]
  (with-open [model (.loadModel (MxModelZoo/SSD) (ProgressBar.))
              predictor (.newPredictor model)]
    (let [predict-result (.predict predictor source)]
      (ImageVisualization/drawBoundingBoxes source predict-result)
      (ImageIO/write source "png" dest))))

(comment
  (detect-image-objects (BufferedImageUtils/fromUrl "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/pose/soccer.png")
                        (io/file "ssd.png")))
