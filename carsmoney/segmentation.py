from .pipeline import Pipeline, register


@register
class Segmentation(Pipeline):
    """This is the segmentation pipeline for stage 1"""
    def load_data(self):
        """TODO(@sundeyichina): Use the segmentation masks that we provide."""
