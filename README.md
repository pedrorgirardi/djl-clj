# djl-clj

## Examples

### Single Shot (Object) Detection

```clojure
(with-open [model (load-model :ssd)
            predictor (predictor model)]
    (let [image (image-from-url (io/resource "soccer.png"))
          detections (predict predictor image)]
      (ImageVisualization/drawBoundingBoxes image detections)
      (ImageIO/write image "png" (File. "ssd.png"))))
```

### Bert QA
```clojure
(with-open [model (load-model {:application Application$NLP/QUESTION_ANSWER
                               :input QAInput
                               :output String
                               :progress (progress-bar)
                               :filter {"backbone" "bert"
                                        "dataset" "book_corpus_wiki_en_uncased"}})
            predictor (predictor model)]
    (let [input (QAInput.
                  "When did BBC Japan start broadcasting?"
                  "BBC Japan was a general entertainment Channel.
                   Which operated between December 2004 and April 2006.
                   It ceased operations after its Japanese distributor folded."
                  384)]
      (predict predictor input)))

;; => "[december, 2004]"
```

### Test

```
$ bin/test
```
