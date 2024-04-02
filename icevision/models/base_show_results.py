__all__ = ["base_show_results"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.visualize import *
from icevision.data import *


def base_show_results(
    predict_fn: callable,
    model: nn.Module,
    dataset: Dataset,
    num_samples: int = 6,
    ncols: int = 3,
    denormalize_fn: Optional[callable] = denormalize_imagenet,
    show: bool = True,
    **kwargs,
) -> None:
    
    predict_kwargs = {k: v for k, v in kwargs.items() if k in predict_fn.__code__.co_varnames}
    draw_sample_kwargs = {k: v for k, v in kwargs.items() if k in draw_sample.__code__.co_varnames}

    records = random.choices(dataset, k=num_samples)
    preds = predict_fn(model, records, **predict_kwargs)

    show_preds(
        preds,
        denormalize_fn=denormalize_fn,
        ncols=ncols,
        show=show,
        **draw_sample_kwargs,
    )
