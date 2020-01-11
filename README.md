NeuralNetworkSimulation
=========================

專案開發時間
--------------

2018/10/09~2019/8/28


摘要
------

碩士論文研究專案輔助用

以Mobilenet為基礎並量化參數，以Verilog實現硬體，軟體模擬運算方式，最後軟硬體輸出資料比對。

利用Quantization-Aware Training Technique and Post-Training Fine-Tuning Quantization 量化參數。

Quantization-Aware Training Technique : Reference DoReFa 
(https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net)

Post-Training Fine-Tuning Quantization : K-means and Linear_transform。

Program summary
-----------------

*之前研究相關不方便公開 : self_mobilenetv2_cifar10.py

*只提供軟體模擬運算程式和部分量化方法。 

*Main program : model_control.py

*主要DL model 主成元件 : average_pooling, batch_normalization, convolution, dense, global_average_pooling, padding, softmax, mobilenetv2_cifar10

*軟硬體輸出答案比對 : data_check.py

*數值轉換 : quantize_valid.py

*數值量化 : quantize_convert.py

