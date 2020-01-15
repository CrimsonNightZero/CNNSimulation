NeuralNetworkSimulation
=========================

專案開發時間
--------------

2018/10/09~2019/8/28


摘要
------

碩士論文研究專案輔助用。

使用輕量化模型 MobileNet，在軟體上針對模型的量化使用Quantization-Aware Training Technique and Post-Training Fine-Tuning Quantization的技術，達到訓練收斂速度的改善以及參數量最小化。在硬體設計考量上,相較於使用浮點數做運算，定點數運算能減少運算複雜度和記憶體儲存空間，這也直接的影響電路的功耗。 MobileNet 硬體加速器的設計上，改善記憶體存取次數，並透過減少 batch normalization 的參數，減少硬體運算複雜度以及參數量,因此所設計的 MobileNet 硬體加速器，結合這些技術的運用，以達到低功耗的表現適合應用於感測器端使用。以Mobilenet為基礎並量化參數，以Verilog實現硬體，軟體模擬運算方式，最後軟硬體輸出資料比對。

Quantization-Aware Training Technique : Reference DoReFa 
(https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net)

Post-Training Fine-Tuning Quantization : K-means and Linear_transform。

Program summary
-----------------

* 之前研究相關不方便公開 : self_mobilenetv2_cifar10.py, Verilog code, DoReFa for mobilenetv2_cifar10 code.

* 只提供軟體模擬運算程式和部分量化方法。 

* Main program : model_control.py

* 主要DL model 主成元件 : average_pooling, batch_normalization, convolution, dense, global_average_pooling, padding, softmax, mobilenetv2_cifar10

* 軟硬體輸出答案比對 : data_check.py

* 數值轉換 : quantize_valid.py

* 數值量化 : quantize_convert.py

* Thesis : https://hdl.handle.net/11296/c48z7a

