import logging
import apache_beam as beam
from apache_beam.metrics.metric import Metrics
from typing import Tuple, Dict, Any, Iterable

class ProcessAndCombineResults(beam.DoFn):
    """
    Processes the co-grouped results containing the original data and the
    NER inference output. It combines them into a single, flat record.
    """
    def process(self, element: Tuple[str, Dict[str, Any]]) -> Iterable[dict]:
        """
        Processes the joined data.

        Args:
            element: A tuple containing the key and a dictionary with two keys:
                     'original': A list with the original data dictionary.
                     'ner': A list with the NER model's prediction result.
        """
        key, data = element
        
        if not data['original'] or not data['ner']:
            Metrics.counter('CombineResults', 'MissingOriginalOrNerData').inc()
            return
            
        original_record = data['original'][0]
        ner_prediction = data['ner'][0]

        extracted_genes = [
            entity['word'] for entity in ner_prediction if entity.get('entity_group') == 'GENE'
        ]
        extracted_diseases = [
            entity['word'] for entity in ner_prediction if entity.get('entity_group') == 'DISEASE'
        ]

        final_record = original_record.copy()
        
        final_record['extracted_genes'] = extracted_genes
        final_record['extracted_diseases'] = extracted_diseases
        
        final_record['is_gene_validated'] = (
            original_record.get('gene_symbol') in extracted_genes
        )
        
        yield final_record