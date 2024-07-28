from dataclasses import dataclass
from PIL import Image


@dataclass
class EasyVQARawElement:
    question: str
    answer: str
    image_id: int
    image_path: int
    image: Image


@dataclass
class EasyVQAElement:
    question: str
    label: str
