"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --input-source "asianboss.mp4" \
"""

import torch
import numpy as np
import cv2

def convert_video(variant, model_path,  input_image, dtype=torch.float32):

    def to_tensor(a):
        return torch.tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) /255

    #
    input_tensor = to_tensor(input_image)

    # load model
    model = MattingNetwork(variant).eval()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    #
    bgr = torch.tensor([120, 255, 155], dtype=dtype).div(255).view(1, 1, 3, 1, 1)

    with torch.no_grad():
        rec = [None] * 4
        # if downsample_ratio is None:
        downsample_ratio = auto_downsample_ratio(*input_tensor.shape[2:])

        input_tensor = input_tensor.unsqueeze(0)  # [B, T, C, H, W]
        fgr, pha, *rec = model(input_tensor, *rec, downsample_ratio)
        com = fgr * pha + bgr * (1 - pha)
        output_img = np.transpose(com[0].data.cpu().numpy(), [0, 2, 3, 1])[0]
        return output_img


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


if __name__ == '__main__':
    import argparse
    from model import MattingNetwork

    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    args = parser.parse_args()
    #用opencv把demo的视频读出来
    cap = cv2.VideoCapture(0)
    # cap.start()
    if cap.isOpened():
        print("open camera success")
        rval, frame = cap.read()
    else:
        print("open camera failed")

    width = 640
    height = 360
    dim = (width, height)
    cv2.namedWindow('raw')
    cv2.namedWindow('cam')

# if vc.isOpened(): # try to get the first frame
#     rval, frame = vc.read()
#     print('Original Dimensions : ',frame.shape)
# else:
#     rval = False

    while True:
        cv2.imshow('raw', cv2.flip(frame, 1))
        ret, frame = cap.read()
        #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        #variant：模型类型，可以写死
        #model_path：模型存储的位置
        #input_img：cv2读到的图片
        #dtype：数据类型，可以写死
        #output_img:输出图片
        output_img = convert_video(
            variant=args.variant,
            model_path=args.checkpoint,
            input_image=frame,
            dtype=torch.float32,
        )
        
        cv2.imshow('cam',output_img)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    
    cv2.destroyWindow("raw")
    cv2.destroyWindow("cam")


