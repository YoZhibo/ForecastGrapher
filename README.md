# ForecastGrapher

Paper Link: [ForecastGrapher: Redefining Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2405.18036)

## Training Scripts

- Place the datasets in the ```./dataset/``` folder (for example，```./dataset/ETTm1.csv```, ```./dataset/PEMS03/PEMS03.npz```). 

- Run the sh file in the `./scripts` folder , using the following command:  `sh ./scripts/ForecastGrapher_***.sh`.

## Model

### Overall Structure

1. Consider each time series as a node and generate a corresponding node embedding. 
2. Employ learnable scalers to partition the node embedding into multiple groups. 
3. Several layers of GFC-GNN are stacked. 
4. Utilize node projection for forecasting.

<div align=center>
<img src="https://github.com/YoZhibo/ForecastGrapher/blob/main/pic/overall_model.jpg" width='60%'> 
</div>




  ### Learnable Scaler and Group Feature Convolution

  An automatic adjustment of feature distributions can be achieved within CNNs. The GFC mechanism enhances the diversity of node embedding distributions: Convoluting the node feature with two distinct kernel lengths results in two distinct distributions.

  

  <div align=center>
  <img src="https://github.com/YoZhibo/ForecastGrapher/blob/main/pic/GFC.jpg" width='50%'> 
  </div>

  


  ## Multivariate Time Series Forecasting Results

  - Comparison with Benchmarks (Avg)

    <div align=center>
    <img src="https://github.com/YoZhibo/ForecastGrapher/blob/main/pic/main_results.jpg" width='85%'> 
    </div>

  - Comparison with advanced GNNs and Naive Method (Avg)

    <div align=center>
    <img src="https://github.com/YoZhibo/ForecastGrapher/blob/main/pic/main_results_gnn.jpg" width='85%'> 
    </div>

  

  ## Acknowledgement

  We appreciate the valuable contributions of the following GitHub.

  - iTransformer (https://github.com/thuml/iTransformer)
  - LTSF-Linear (https://github.com/cure-lab/LTSF-Linear)
  - Time-Series-Library (https://github.com/thuml/Time-Series-Library)
  - FourierGNN (https://github.com/aikunyi/FourierGNN)
  - StemGNN (https://github.com/microsoft/StemGNN)

