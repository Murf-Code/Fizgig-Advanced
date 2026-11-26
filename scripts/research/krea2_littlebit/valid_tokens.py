"""Sample valid image/text rows before observing gradients."""
import torch


def valid_rows(image_tokens,text_mask,rows=16):
    if text_mask.ndim!=2 or text_mask.shape[0]!=1 or text_mask.dtype!=torch.bool:
        raise ValueError("LittleBit capture requires one image and a boolean text mask")
    if image_tokens<1 or rows<1:
        raise ValueError("invalid sample size")
    txt=text_mask[0].nonzero(as_tuple=True)[0]+image_tokens
    image=torch.arange(image_tokens,device=text_mask.device)
    ni=min(len(image),(rows+1)//2)
    nt=min(len(txt),rows-ni)
    ni=min(len(image),rows-nt)
    def pick(v,n):
        if not n: return v[:0]
        return v[torch.linspace(0,len(v)-1,n,device=v.device).long()]
    idx=torch.cat((pick(image,ni),pick(txt,nt)))
    return idx,{"image_tokens":image_tokens,"valid_text_tokens":len(txt),
                "padding_tokens":text_mask.shape[1]-len(txt),
                "sampled_image":ni,"sampled_text":nt,"indices":idx.cpu().tolist(),
                "sequence_tokens":image_tokens+text_mask.shape[1]}
