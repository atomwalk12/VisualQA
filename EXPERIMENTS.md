# Experiments


## Easy VQA Dataset

### Fine tune a classifier
To fine tune using a classification head use:
```python
> python finetune.py --model blip2-classifier --dataset easy-vqa --train-split 'train' --val-split 'val'
```

#### Test the classifier
```python
> python test.py --model blip2-classifier --dataset easy-vqa --test-split 'test[:500]'
```
#### Visualization

##### Generate a confusion matrix
To generate the confusion matrix run:
```python
> python visualization.py --model blip2-classifier --dataset easy-vqa --metric confusion-matrix --split 'train[:1000]'
```

### Fine tune a token generator
To fine tune only the generator without a classification head use:
```python
> python finetune.py --model blip2-generator --dataset easy-vqa --train-split 'train[:1000]' --val-split 'val[:200]'
```

There are no confusion matrices here, but multiple assessment metrics are used: perplexity, token_level_accuracy, adapted_bleu_score, entropy_of_predictions, top_5_accuracy.

#### Test the trained model using:
```python
> python test.py --model blip2-generator --dataset easy-vqa --test-split 'test[:1000]'
```

#### Visualization
##### UMap

To visualize the embeddings in a 2-dimensional graph using UMAP write the following:
```python
> python visualization.py --model blip2-classifier --dataset easy-vqa --metric umap --split 'test[:500]'
```


## Daquar dataset
The split count must be greater or equal to the number of classes 53 after filtering, 582 a priori.
### Fine tune a classifier
To fine tune using a classification head use:
```python
> python finetune.py --model blip2-classifier --dataset daquar --train-split 'train[:1000]' --val-split 'val[:200]'
```

#### Test the classifier
```python
> python test.py --model blip2-classifier --dataset daquar --test-split 'test'
```
#### Visualization

##### Generate a confusion matrix:
To generate the confusion matrix run:
```python
> python visualization.py --model blip2-classifier --dataset daquar --metric confusion-matrix --split 'train[:1000]'
``` 

### Fine tune a token generator
To fine tune only the generator without a classification head use:
```python
> python finetune.py --model blip2-generator --dataset daquar --train-split 'train[:1000]' --val-split 'val[:200]'
```

#### Test the trained model using:
```python
> python test.py --model blip2-generator --dataset daquar --test-split 'test[:1000]'
```

#### Visualization
##### UMap

To visualize the embeddings in a 2-dimensional graph using UMAP write the following:
```python
> python visualization.py --model blip2-classifier --dataset daquar --metric umap --split 'test[:300]'
```