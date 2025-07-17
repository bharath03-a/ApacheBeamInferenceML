import json
import logging

import apache_beam as beam
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.huggingface_inference import HuggingFaceModelHandlerKeyedTensor

from utils.pipeline_options import NerPipelineOptions
from utils.parsing import (ParseClinVarTxt, ValidateAndPrepareVariant, 
                              SelectAndRenameFields, FanoutPrompts)
from utils.inference import ProcessAndCombineResults


def run():
    """Constructs and runs the entire NER A/B testing pipeline."""
    logging.getLogger().setLevel(logging.INFO)
    
    options = NerPipelineOptions()

    model_handler = HuggingFaceModelHandlerKeyedTensor(
        model_uri=options.model_name,
        model_class='AutoModelForTokenClassification',
        tokenizer_class='AutoTokenizer',
        task='ner',
        batch_size=options.batch_size
    )

    with beam.Pipeline(options=options) as p:
        # Step 1: Read, parse, validate, and fan out data for A/B testing
        fanned_out_data = (
            p
            | "ReadInputFile" >> beam.io.ReadFromText(options.input)
            | "ParseTextToDict" >> beam.ParDo(ParseClinVarTxt())
            | "ValidateAndPreparePrompts" >> beam.ParDo(ValidateAndPrepareVariant())
            | "SelectAndRename" >> beam.ParDo(SelectAndRenameFields())
            | "FanoutForABTesting" >> beam.ParDo(FanoutPrompts())
        )

        # Step 2 (Branch A): Prepare data for inference and run the model
        inference_results = (
            fanned_out_data
            | "MapKeyAndTextForInference" >> beam.Map(lambda x: (x['allele_id'], x['inference_text']))
            | "RunNERInference" >> RunInference(model_handler)
            # Output is now PCollection[(key, ner_result)]
        )

        # Step 3 (Branch B): Key the original fanned-out data for joining
        keyed_original_data = (
            fanned_out_data
            | "MapKeyToOriginalData" >> beam.Map(lambda x: (x['allele_id'], x))
        )

        # Step 4: Join the inference results back with the original data
        combined_data = (
            {'original': keyed_original_data, 'ner': inference_results}
            | "CombineOriginalAndNER" >> beam.CoGroupByKey()
        )

        # Step 5: Process the combined data and format for output
        final_output = (
            combined_data
            | "ProcessCombinedResults" >> beam.ParDo(ProcessAndCombineResults())
            | "FormatAsJson" >> beam.Map(json.dumps)
        )
        
        # Step 6: Write the final JSON records to a file
        final_output | "WriteOutput" >> beam.io.WriteToText(
            options.output_path, file_name_suffix='.jsonl.gz'
        )

if __name__ == '__main__':
    run()