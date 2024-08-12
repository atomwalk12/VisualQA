# Experiments


## Easy VQA Dataset
To fine tune the model run:
```python
> python finetune.py --model blip2-classifier --dataset easy-vqa --train-split 'train[:80]' --val-split 'train[:80]'
```

To check the confusion matrix run:
```python
> python visualization.py --model blip2-classifier --dataset easy-vqa --metric confusion-matrix --split 'train[:80]'
```