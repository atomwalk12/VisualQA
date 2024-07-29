# Create the dataset

```python
> mkdir -p data/easy-vqa data/models
> python finetune_blip2.py --model blip2 --dataset easy-vqa --task process-data --output-dir data/easy-vqa --model-dir data/models
```

```python
> mkdir -p data/easy-vqa data/models
> python finetune_blip2.py --model blip2 --dataset easy-vqa --task fine-tune --output-dir data/easy-vqa --model-dir data/models
```
