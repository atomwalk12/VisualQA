from enum import StrEnum


class Repositories(StrEnum):
    VQAGenerationEasyVQA = "atomwalk12/blip2-easyvqa-gen"
    VQAGenerationDaquar = "atomwalk12/blip2-daquar-gen"
    VQAClassificationEasyVQA = "atomwalk12/blip2-easyvqa-classifier"
    VQAClassificationDaquar = "atomwalk12/blip2-daquar-classifier"
