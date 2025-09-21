import transformers
from watermark_utils import in_give_score_bys,get_detector_path,get_watermark


def  give_watermark_score(text,model_name,device):
    watermark = get_watermark(model_name)
    detector_path = get_detector_path(model_name)
    score = in_give_score_bys(text,model_name,watermark,device,detector_path)
    return score
