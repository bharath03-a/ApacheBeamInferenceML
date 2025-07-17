import logging
import apache_beam as beam
from apache_beam.metrics.metric import Metrics
from typing import Iterable

class ParseClinVarTxt(beam.DoFn):
    """
    Parses a single row from ClinVar's variant_summary.txt.
    This class remains as-is, as it correctly parses all columns.
    """
    def process(self, element: str) -> Iterable[dict]:
        if element.startswith("#"):
            return

        columns = [
            "#AlleleID", "Type", "Name", "GeneID", "GeneSymbol", "HGNC_ID", "ClinicalSignificance", "ClinSigSimple", "LastEvaluated",
            "RS# (dbSNP)", "nsv/esv (dbVar)", "RCVaccession", "PhenotypeIDS", "PhenotypeList", "Origin", "OriginSimple", "Assembly",
            "ChromosomeAccession", "Chromosome", "Start", "Stop", "ReferenceAllele", "AlternateAllele", "Cytogenetic", "ReviewStatus",
            "NumberSubmitters", "Guidelines", "TestedInGTR", "OtherIDs", "SubmitterCategories", "VariationID", "PositionVCF", "ReferenceAlleleVCF",
            "AlternateAlleleVCF", "SomaticClinicalImpact", "SomaticClinicalImpactLastEvaluated", "ReviewStatusClinicalImpact", "Oncogenicity",
            "OncogenicityLastEvaluated", "ReviewStatusOncogenicity", "SCVsForAggregateGermlineClassification", "SCVsForAggregateSomaticClinicalImpact", "SCVsForAggregateOncogenicityClassification"
        ]
        try:
            fields = element.strip().split("\t")
            if len(fields) != len(columns):
                Metrics.counter('ParseErrors', 'ColumnCountMismatch').inc()
                return
            data_dict = dict(zip(columns, fields))
            yield data_dict
        except Exception as e:
            logging.error(f"Error parsing row: {element}, error: {e}")
            Metrics.counter('ParseErrors', 'Exception').inc()
            return

class ValidateAndPrepareVariant(beam.DoFn):
    """
    Validates parsed data and generates two distinct prompts for A/B testing.
    """
    def process(self, element: dict) -> Iterable[dict]:
        required_fields = ["#AlleleID", "Name", "GeneSymbol", "PhenotypeList", "ClinicalSignificance", "ReviewStatus", "Type"]
        
        for field in required_fields:
            if not element.get(field) or element.get(field) == '-':
                Metrics.counter('Validation', f'MissingOrEmpty_{field}').inc()
                return

        try:
            if element.get("Start"): element["Start"] = int(element["Start"])
            if element.get("Stop"): element["Stop"] = int(element["Stop"])
        except (ValueError, TypeError):
            Metrics.counter('Validation', 'InvalidPositionFormat').inc()
            return

        # Prompt A: "Natural Sentence" - simple and direct.
        element["inference_text_A"] = (
            f"{element['Name']}, a {element['Type']} variant in the {element['GeneSymbol']} gene, "
            f"is associated with {element['PhenotypeList']}."
        )

        # Prompt B: "Clinical Context" - richer, structured prompt.
        element["inference_text_B"] = (
            f"Analyze the following genetic variant based on its clinical summary.\n\n"
            f"Variant Details:\n"
            f"- Name: {element['Name']}\n"
            f"- Gene: {element['GeneSymbol']}\n"
            f"- Type: {element['Type']}\n"
            f"- Assessed Significance: {element['ClinicalSignificance']} (Review Status: {element['ReviewStatus']})\n"
            f"- Known Associated Conditions: {element['PhenotypeList']}"
        )

        Metrics.counter('Validation', 'ValidRecords').inc()
        yield element

class SelectAndRenameFields(beam.DoFn):
    """
    Selects an expanded subset of essential fields and creates a clean record.
    This now passes through the two prompt fields created previously.
    """
    def process(self, element: dict) -> Iterable[dict]:
        fields_to_keep = {
            'allele_id': '#AlleleID',
            'rcv_accession': 'RCVaccession',
            'variant_name': 'Name',
            'variant_type': 'Type',
            'gene_symbol': 'GeneSymbol',
            'clinical_significance': 'ClinicalSignificance',
            'review_status': 'ReviewStatus',
            'phenotypes_text': 'PhenotypeList',
            'phenotypes_ids': 'PhenotypeIDS',
            'origin': 'OriginSimple',
            'last_evaluated': 'LastEvaluated',
            'assembly': 'Assembly',
            'chromosome': 'Chromosome',
            'start_position': 'Start',
            'ref_allele': 'ReferenceAlleleVCF',
            'alt_allele': 'AlternateAlleleVCF',
            # Pass through the two prompt fields for the next step
            'inference_text_A': 'inference_text_A',
            'inference_text_B': 'inference_text_B'
        }
        try:
            trimmed_record = {
                new_name: element.get(old_name) for new_name, old_name in fields_to_keep.items()
            }
            yield trimmed_record
        except KeyError as e:
            Metrics.counter('SelectFields', f'MissingCriticalFieldError_{e}').inc()

class FanoutPrompts(beam.DoFn):
    """
    Takes a single record and creates two separate records for A/B testing,
    one for each prompt type.
    """
    def process(self, element: dict) -> Iterable[dict]:
        base_record = element.copy()
        prompt_a_text = base_record.pop("inference_text_A")
        prompt_b_text = base_record.pop("inference_text_B")

        record_a = base_record.copy()
        record_a["prompt_type"] = "A_NaturalSentence"
        record_a["inference_text"] = prompt_a_text
        yield record_a

        record_b = base_record.copy()
        record_b["prompt_type"] = "B_ClinicalContext"
        record_b["inference_text"] = prompt_b_text
        yield record_b