import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from apache_beam.ml.inference.base import RunInference, ModelHandler
from apache_beam.ml.inference.base import PredictionResult, KeyedModelHandler
from apache_beam.ml.inference.huggingface_inference import HuggingFacePipelineModelHandler
from apache_beam.ml.inference.huggingface_inference import HuggingFaceModelHandlerKeyedTensor
from apache_beam.ml.inference.huggingface_inference import HuggingFaceModelHandlerTensor
from apache_beam.ml.inference.huggingface_inference import PipelineTask

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Iterable, Any

class HuggingFaceModelHandler(ModelHandler[str, str, Any]):
    """
    Custom Model Handler that loads and runs Hugging Face transformer models.
    """
    def __init__(self, model_uri: str = 'distilgpt2', **kwargs):
        """
        Initializes the model handler.

        Args:
            model_uri: The Hugging Face model identifier.
            kwargs: Additional arguments for the model's generate method.
        """
        self._model_uri = model_uri
        self._kwargs = kwargs
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None

    def load_model(self):
        """Loads the model and tokenizer directly from Hugging Face."""
        # Load the model and move it to the configured device
        model = AutoModelForCausalLM.from_pretrained(self._model_uri)
        model.to(self._device)
        model.eval() # Set model to evaluation mode

        # Load the tokenizer and set a padding token if one doesn't exist
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_uri)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return model

    def run_inference(
        self,
        batch: Iterable[str],
        model: Any,
        inference_args: dict = None
    ) -> Iterable[str]:
        """
        Runs inference on a batch of text prompts.
        """
        # Tokenize the batch of prompts
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )

        # Move tensors to the correct device
        inputs = {key: val.to(self._device) for key, val in inputs.items()}

        # Generate text using the model
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                **self._kwargs
            )

        # Decode the generated sequences and yield them
        for output in outputs:
            yield self.tokenizer.decode(output, skip_special_tokens=True)

# --- Main Pipeline ---
def main():
    prompts = [
        "The weather in Rochester, MN is usually",
        "The capital of the United States is",
        "Apache Beam is a great tool for",
    ]

    model_handler = HuggingFaceModelHandler(model_uri='distilgpt2')

    with beam.Pipeline(options=PipelineOptions()) as p:
        (
            p
            | "CreatePrompts" >> beam.Create(prompts)
            | "RunInference" >> RunInference(model_handler)
            | "PrintResults" >> beam.Map(print)
        )

if __name__ == "__main__":
    main()

# interesting so here we created a custome model handler that loads the model and tokenizer from hugging face and runs inference on a batch of text prompts.
# we can use this model handler to run inference on a batch of text prompts.
# we can also use this model handler to run inference on a single text prompt.
# so we can either use a custom model handler or use the beam provided model handlers.
# the beam provided model handlers are HuggingFaceModelHandlerKeyedTensor and HuggingFaceModelHandlerTensor.
# the HuggingFaceModelHandlerKeyedTensor is used when the input is a keyed tensor.
# the HuggingFaceModelHandlerTensor is used when the input is a tensor.
# the HuggingFacePipelineModelHandler is used when the input is a pipeline.
# the HuggingFaceModelHandler is used when the input is a model.

# so I now want to basically find a good use case to implement beam ml inference.
# Use case:
# 1 - Batch Summarization of Server Logs
# 2 - Genomics: Large-Scale Variant Calling and Annotation
# 3 - Quantitative Finance: Market Signal Generation from Unstructured Data