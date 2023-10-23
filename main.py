from PIL import Image
import numpy as np
from StreamManagerApi import StreamManagerApi, \
    MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/shufflenetv2plus.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    img = Image.open("test.JPEG").convert('RGB')
    input_size = 256
    if img.height <= img.width:
        ratio = input_size / img.height
        w_size = int(img.width * ratio)
        img = img.resize((w_size, input_size), Image.BILINEAR)
    else:
        ratio = input_size / img.width
        h_size = int(img.height * ratio)
        img = img.resize((input_size, h_size), Image.BILINEAR)

    img = np.array(img, dtype=np.float32)
    out_width = 224
    out_height = 224
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]

    img = img / 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img[..., 0] = (img[..., 0] - mean[0]) / std[0]
    img[..., 1] = (img[..., 1] - mean[1]) / std[1]
    img[..., 2] = (img[..., 2] - mean[2]) / std[2]

    img = img.transpose(2, 0, 1)   # HWC -> CHW

    vision_list = MxpiDataType.MxpiVisionList()
    vision_vec = vision_list.visionVec.add()
    vision_vec.visionInfo.format = 0
    vision_vec.visionInfo.width = 224
    vision_vec.visionInfo.height = 224
    vision_vec.visionInfo.widthAligned = 224
    vision_vec.visionInfo.heightAligned = 224

    vision_vec.visionData.memType = 0
    vision_vec.visionData.dataStr = img.tobytes()
    vision_vec.visionData.dataSize = len(img)

    protobuf = MxProtobufIn()
    protobuf.key = b"appsrc0"
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = vision_list.SerializeToString()
    protobuf_vec = InProtobufVector()

    protobuf_vec.push_back(protobuf)

    # Inputs data to a specified stream based on streamName.
    inPluginId = 0
    uniqueId = streamManagerApi.SendProtobuf(b'im_shufflenetv2plus', inPluginId, protobuf_vec)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = streamManagerApi.GetResult(b'im_shufflenetv2plus', uniqueId)
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        exit()

    # print the infer result
    print(inferResult.data.decode())

    # destroy streams
    streamManagerApi.DestroyAllStreams()
