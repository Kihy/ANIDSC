



METRICS = [
    "detection_rate",
    "lower_quartile_score",
    "upper_quartile_score",
    "soft_min_score",
    "soft_max_score",
    "median_score",
    "median_threshold",
    "pos_count",
    "batch_size",
]

CDD_METRICS=METRICS+["pool_size", "drift_level"]

FE_TEMPLATE="""
components:
  data_source:
    class: PacketReader
    attrs: 
      dataset_name: {}
      file_name: {}    
  feature_extractor:
    class: AfterImage
  feature_buffer:
    class: TabularFeatureBuffer
"""


def get_template(template_type):
    if template_type=="feature_extraction":
        return FE_TEMPLATE