# Main Experiments
## Create the dataset

```python
> mkdir -p data/easy-vqa data/models
> python finetune_blip2.py --model blip2 --dataset easy-vqa --task process-data --data-dir data/easy-vqa --model-dir data/models
```

## Run fine-tuning
```python
> mkdir -p data/easy-vqa data/models
> python finetune_blip2.py --model blip2 --dataset easy-vqa --task fine-tune --data-dir data/easy-vqa --model-dir data/models
```

# Instructions

To generate a fixed number of examples in training in validation datasets use the flags --train train[:30] --val val[:10].

```python
> mkdir -p data/easy-vqa data/models
> python finetune_blip2.py --model blip2 --dataset easy-vqa --task process-data --data-dir data/easy-vqa --model-dir data/models --val "val[:15]" --train "train[:30]"
```

To performed training/validation for a fixed number of epochs.

```python
> mkdir -p data/easy-vqa data/models
> python finetune_blip2.py --model blip2 --dataset easy-vqa --task fine-tune --data-dir data/easy-vqa --model-dir data/models --val "val[:15]" --train "train[:30]" --limit-train-batches 200 --limit-val-batches 50
```
