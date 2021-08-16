import cProfile
import os
from logging import getLogger

logger = getLogger(__name__)


def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        enable_profile = int(os.getenv("PROFILE", 1))
        if enable_profile:
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats()
        else:
            return func(*args, **kwargs)

    return profiled_func

def log_decorator(endpoint: str="/", logger=logger):
    def _log_decorator(func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            job_id = kwargs.get("job_id")
            data = kwargs.get("data")
            prediction = kwargs.get("prediction")
            prediction_elapsed = kwargs.get("prediction_elapsed")
            is_outlier = kwargs.get("is_outlier")
            outlier_score = kwargs.get("outlier_score")
            outlier_elapsed = kwargs.get("outlier_elapsed")

            logger.info(
                f"[{endpoint}] [{job_id}] [{data}] [{prediction}]] [{prediction_elapsed}] [{is_outlier}]] [{outlier_score}] [{outlier_elapsed}]"
            )
            return res 

        return wrapper
        
    return _log_decorator
