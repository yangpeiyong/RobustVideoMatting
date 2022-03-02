"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --input-source "asianboss.mp4" \
"""

import tensorflow as tf
import numpy as np
import cv2

def convert_video(model, input_image):


        # out = model([src, *rec, downsample_ratio])
        # fgr, pha, *rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']

    # img_bgr = cv2.imread(filename='/path/to/file')
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # input_image = input_image / 255
    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
    input_tensor = input_tensor / 255.0
    # input_tensor = to_tensor(input_image)

    print(input_tensor.shape)

    input_tensor = tf.expand_dims(input_tensor, axis=0)

    print(input_tensor.shape)

    rec = [ tf.constant(0.) ] * 4         # Initial recurrent states.
    downsample_ratio = tf.constant(0.5)  # Adjust based on your video.

        # if downsample_ratio is None:
    # downsample_ratio = auto_downsample_ratio(*input_tensor.shape[2:])
    # bgr = torch.tensor([120, 255, 155], dtype=dtype).div(255).view(1, 1, 3, 1, 1)

    # input_tensor = input_tensor.unsqueeze(0)  # [B, T, C, H, W]

    # fgr, pha, *rec = model([input_tensor, *rec, downsample_ratio])

    out = model([input_tensor, *rec, downsample_ratio])
    fgr, pha, *rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']

    #input_image = tf.convert_to_tensor(input_image, dtype=tf.float32) / 255.0

    com = fgr * pha

    print(fgr.shape)
    print(pha.shape)
    print(com.shape)
    com = tf.squeeze(com)
    print(com.shape)
    #output_img = np.transpose(com[0].data.cpu().numpy(), [0, 2, 3, 1])[0]
    output_img = np.array(com)
    return output_img


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


if __name__ == '__main__':


    model = tf.keras.models.load_model('rvm_mobilenetv3_tf')
    model = tf.function(model)

    rec = [ tf.constant(0.) ] * 4         # Initial recurrent states.
    downsample_ratio = tf.constant(0.25)  # Adjust based on your video.


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
            model=model,
            input_image=frame
        )
        
        cv2.imshow('cam',output_img)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    
    cv2.destroyWindow("raw")
    cv2.destroyWindow("cam")


