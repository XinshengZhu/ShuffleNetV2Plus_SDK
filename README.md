# ShuffleNetV2Plus

CANN 5.0.4
SDK版本：21.0.4
固件驱动 21.0.4

# 自检报告

## ModelArts训练
### 正确输出了pth、onnx模型文件
```bash
bash run_train.sh [/path/to/code/in/obs] 'code/modelarts/train_start.py' '/tmp/log/training.log' --data_url=[path/to/data/in/obs] --train_url=[/path/to/output/in/obs] --epochs=1 --batch-size=4
```
### 验收结果： OK 
### 备注： 目标输出结果无误

## 模型转换
```bash
cd /infer/convert
bash convert_om.sh ../data/model/shufflenetv2plus_npu.onnx ../data/model/shufflenetv2plus_npu
```

## SDK推理
### 运行SDK推理：
```bash
cd /infer/sdk
bash run.sh ../data/input/imagenet/val/ sdk_pred_result.txt
```
### 测试推理精度：
```bash
python3 ../util/task_metric.py sdk_pred_result.txt ../data/config/val_label.txt sdk_pred_result.acc.json 5
```
### 验收结果： OK 
### 备注： 输出了正确的推理结果、推理精度达标

## MxBase推理
### 编译可执行程序：
```bash
cd /infer/mxbase
bash build.sh
```
### 预处理数据集：
```bash
python3 ../util/preprocess.py ../data/input/imagenet/val/ binfile
```
### 运行推理程序：
```bash
./build/shufflenetv2plus ./binfile/
```
### 测试推理精度：
```bash
python3 ../util/task_metric.py mx_pred_result.txt ../data/config/val_label.txt sdk_pred_result.acc.json 5
```
### 验收结果： OK 
### 备注： 输出了正确的推理结果、推理精度达标
