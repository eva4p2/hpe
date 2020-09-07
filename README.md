# Monocular Human Pose Estimation

This approach of Human Pose Estimation is the simplest of others arrived since 2016 to till date. The model is based on RESNET with a few Deconvoltions added on top of convolutions. MSRA (Microsoft Research Asia) team have done a very immpressive work on COCO and MPII Dataset. Follow the github - https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/README.md

![Image](https://github.com/eva4p2/hpe/blob/master/MPII-hpe.png)

In this assignment the focus is more on MPII Dataset with 256x256_pose_resent_50 model. Below is the image in which model level comparison is done at the architecture level with hourglass and CPN. Each model is an encoder-decoder model with 8x upssampling. Monocular Human Pose Estimation uses Resnet50 as the backbone model on which deconvolutions are added (at C5 block in Resnet50) to geneate 16 images of size 64x64. Hence the output of the model is an array of shape 16x64x64. Each array is the representation of joint detected on an input pose image.

![Image](https://github.com/eva4p2/hpe/blob/master/model.png)

This assignment is based on Discriminative, bottom-up and Multistage mothod.

![Image](https://github.com/eva4p2/hpe/blob/master/Methods-hpe.png)

A very important step of transporting a model written in one framework to another framework is done using ONNX. OONX is Open Neural Network Exchange through which one can transport a model developed in pytorch to a keras/caffe model and//or vice versa. ONNX runtime can also be used to do the inferencing without model conversion to any other framework model.

Note - This model is to be deployed on AWS lambda using serverless framework but the limitation for AWS lambda is 250 zipped contect.Hence, keeping a pretrained model either in pytorch or ONNX is a challenge. To resolve this ONNX quantization is used to reduce the size of the pretrained model.
```

# from onnxruntime.quantization import quantize
# from onnxruntime.quantization import QuantizationMode

# # Quantize
# quantized_model = quantize(model_onnx, quantization_mode=QuantizationMode.IntegerOps)
 
# # Save the quantized model
# onnx.save(quantized_model, 'quantized_model_1.onnx')

```

Steps to do conversion/transportation is perfomed in Session5-HPE.ipynb
### Input Image
![Image](https://github.com/eva4p2/hpe/blob/master/pose.jpg)
### 1. load the pytorch model state dict for Monocular Human Pose Estimation
### 2. check for keypoint predictions
![Image](https://github.com/eva4p2/hpe/blob/master/pose-with-keypoints.png)
### 3. use pypi onnx to save the model as .onnx file
### 4. (optional) load onnx model and further saved it as quantized model
### 5. infer via quantized model and join keypoints
### Output Image
![Image](https://github.com/eva4p2/hpe/blob/master/pose-hpe.png)



References- 
  All the models are stored in AWS S3 public bucket -
  1. https://sandhu-tsai-models-s5.s3.ap-south-1.amazonaws.com/pose_resnet_50_256x256.pth.tar
  2. https://sandhu-tsai-models-s5.s3.ap-south-1.amazonaws.com/simple_pose_estimation_1.onnx
  3. https://sandhu-tsai-models-s5.s3.ap-south-1.amazonaws.com/quantized_model_1.onnx


