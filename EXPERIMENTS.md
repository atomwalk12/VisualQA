# Main Experiments
## Create the Easy-VQA dataset

```python
> mkdir -p data/easy-vqa data/models
> python process.py --model blip2 --dataset easy-vqa --train "train[:1000]" --val "val[:120]"
```

## Fine tune Blip 2 using lightning
```python
> mkdir -p data/easy-vqa data/models
> python finetune.py --model blip2 --dataset easy-vqa --train "train[:1000]" --val "val[:120]" --use-lightning
```

## Fine tune Blip2 using simple pytorch loop
```python
> mkdir -p data/easy-vqa data/models
> python finetune.py --model blip2 --dataset easy-vqa --train "train[:1000]" --val "val[:120]"
```
