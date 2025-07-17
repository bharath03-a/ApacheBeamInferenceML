from apache_beam.options.pipeline_options import PipelineOptions

class NerPipelineOptions(PipelineOptions):
    """Custom pipeline options for the ClincalVar NER pipeline.
    """
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            "--input_file",
            type=str,
            help="The path to the input file.",
        )

        parser.add_argument(
            "--output_file",
            required=True,
            type=str,
            help="The path to the output file.",
        )

        parser.add_argument(
            "--model_name",
            required=True,
            type=str,
            help="The name of the model.",
        )

        parser.add_argument(
            "--bq_dataset",
            required=True,
            type=str,
            help="The name of the BigQuery dataset.",
        )

        parser.add_argument(
            "--bq_table",
            required=True,
            type=str,
            help="The name of the BigQuery table.",
        )

        parser.add_argument(
            '--staging_location',
            required=True,
            type=str,
            help='Cloud storage path for staging pipeline code and dependencies.'
        )

        parser.add_argument(
            '--runner',
            required=True,
            type=str,
            help='The runner to use for the pipeline.'
        )
        parser.add_argument(
            '--temp_location',
            required=True,
            type=str,
            help='Cloud storage path for storing temporary files.'
        )