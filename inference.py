import warnings

import torch
from peft.utils.save_and_load import set_peft_model_state_dict

from lib.representations import DatasetFactory, ModelFactory
from lib.types import CustomDataset
from lib.visualization import show_images_with_captions


def show_examples(
    count,
    seed,
    shuffle=True,
    model_path="BestLoss.bin",
    model_name="blip2",
    ds_name="easy-vqa",
):
    # Load the validation set
    val_args = {
        "split": "val[:30]",
        "load_raw": False,
        "prepare_for_training": False,
    }

    # Load the model and processor
    model, processor = ModelFactory.get_models(model_name, apply_qlora=True)

    # Load the training and validation sets
    _, val_ds = DatasetFactory.create_dataset(ds_name, None, val_args)
    val_ds: CustomDataset = val_ds.load()

    if shuffle:
        val_ds.shuffle(seed=seed)

    # Load best model
    adapter_weights = torch.load(model_path)
    set_peft_model_state_dict(model, adapter_weights)

    # Generate responses for the first 5 elements
    elements = val_ds[:count]
    responses = []
    for image, prompt in zip(elements["image"], elements["prompt"]):
        # Generate the input_ids, image_ids, etc.
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            device="cuda", dtype=torch.bfloat16
        )

        # Generate outputs
        generated_ids = model.generate(**inputs, max_new_tokens=5)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        responses.append(f"{prompt}\n{generated_text[0].strip()}")

    # Show images with captions
    show_images_with_captions(images_or_paths=val_ds[:count]["image"], captions=responses)


if __name__ == "__main__":
    # Disable warning
    warnings.filterwarnings("ignore")

    show_examples(count=12, seed=2024)
