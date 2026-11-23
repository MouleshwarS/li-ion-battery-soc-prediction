# Lithium-Ion Battery SoC Prediction Using Deep Learning
Designed Vanilla RNN, LSTM and GRU models to predict the state of charge of a lithium-ion battery used in an EV.

<details>
<summary><b>Click to expand the repository structure</b></summary>

<br>

```text
li-ion-battery-soc-prediction/
├── train_test/
│   ├── GRU_SoC_Layers_Final.ipynb      # Optimal No. of Hidden Layer(s) for GRU
│   ├── GRU_SoC_Nodes_Final.ipynb       # Optimal No. of HL Nodes for GRU
│   ├── LSTM_SoC_Layers_Final.ipynb     # Optimal No. of Hidden Layer(s) for LSTM
│   ├── LSTM_SoC_Nodes_Final.ipynb      # Optimal No. of HL Nodes for LSTM
│   ├── RNN_SoC_Layers_Final.ipynb      # Optimal No. of Hidden Layer(s) for RNN
│   └── RNN_SoC_Nodes_Final.ipynb       # Optimal No. of HL Nodes for RNN
├── EDA_CategoryA.ipynb                 # EDA for Category A Trips
├── EDA_CategoryB.ipynb                 # EDA for Category B Trips
└── README.md                           # Project Description
```

</details>

## Dataset Description
The dataset contains data for driving trips that have been made using a BMW i3 EV. It has been originally sourced from IEEE Dataport, but published in Kaggle.
* Title: BATTERY AND HEATING DATA IN REAL DRIVING CYCLES
* URL: https://www.kaggle.com/datasets/atechnohazard/battery-and-heating-data-in-real-driving-cycles

<table>
  <thead>
    <tr>
      <th align="center" colspan="2">Generic Description of Dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Number of trips</td>
      <td align="center"><code>70</code></td>
    </tr>
    <tr>
      <td align="center">Number of categories</td>
      <td align="center"><code>2</code></td>
    </tr>
    <tr>
      <td align="center">Number of trips in category A (summer)</td>
      <td align="center"><code>32</code></td>
    </tr>
    <tr>
      <td align="center">Number of trips in category B (winter)</td>
      <td align="center"><code>38</code></td>
    </tr>
    <tr>
      <td align="center">Total number of samples</td>
      <td align="center"><code>10,94,793</code></td>
    </tr>
    <tr>
      <td align="center">Number of samples in category A</td>
      <td align="center"><code>4,67,701</code></td>
    </tr>
    <tr>
      <td align="center">Number of samples in category B</td>
      <td align="center"><code>6,27,092</code></td>
    </tr>
  </tbody>
</table>

### Training and Test Datasets
The training dataset has combined data from both categories, sorted in decreasing order of battery SoC.

<table>
  <thead>
    <tr>
      <th align="center" colspan="2">Generic Description of Training Dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Number of trips</td>
      <td align="center"><code>35</code></td>
    </tr>
    <tr>
      <td align="center">Number of trips in category A (summer)</td>
      <td align="center"><code>16</code></td>
    </tr>
    <tr>
      <td align="center">Number of trips in category B (winter)</td>
      <td align="center"><code>19</code></td>
    </tr>
    <tr>
      <td align="center">Total number of samples</td>
      <td align="center"><code>5,38,214</code></td>
    </tr>
    <tr>
      <td align="center">Number of samples in training subset</td>
      <td align="center"><code>4,30,572</code> (80%)</td>
    </tr>
    <tr>
      <td align="center">Number of samples in validation subset</td>
      <td align="center"><code>1,07,642</code> (20%)</td>
    </tr>
    <tr>
      <td align="center">Input features</td>
      <td align="center"><code>3</code> (Battery Voltage, Battery Current,<br>Battery Temperature)</td>
    </tr>
    <tr>
      <td align="center">Target variable</td>
      <td align="center">Battery SoC</td>
    </tr>
  </tbody>
</table>

<br>

On the other hand, the test dataset contains data for `12` unseen trips, `6` each from category A and B. To normalize the training, validation and test datasets, min-max normalization has been used, with a scaling range of `[-1, 1]`.

## Vanilla RNN
The SimpleRNN and Dense layers of the Sequential module of the Keras library in Python were used to implement the Vanilla RNN.
* The activation functions used are as follows:
  * **Tanh or hyperbolic tangent** function for the **hidden layer(s)**
  * **Linear function** for the **output layer**
* **Mean squared error (MSE)** was used as the loss function as well as a performance metric
* Along with MSE, **R-squared (R2)** score was used as a performance metric
* **Stochastic gradient descent (SGD)** with **`0.025`** as the learning rate, was used as the optimizer

The optimal configuration for the Vanilla RNN model was determined using a trial and error approach, in which different configurations were trained for `100` epochs with `512` samples per batch, and their performance metrics were analysed. The configuration with the least possible MSE as well as highest possible R2 score was chosen as the optimal one.

## LSTM
The LSTM and Dense layers of the Sequential module of the Keras library in Python were used to implement the long short-term memory network. The activation functions, loss function, optimizer, learning rate and the performance metrics used are similar to the Vanilla RNN implementation, with the optimal configuration for the LSTM model also being determined in a similar fashion.

## GRU
The GRU and Dense layers of the Sequential module of the Keras library in Python were used to implement the gated recurrent unit network. The activation functions, loss function, optimizer, learning rate and the performance metrics used are similar to the Vanilla RNN implementation, with the optimal configuration for the GRU model also being determined in a similar fashion.

## Model Testing/Validation
The performance of the three optimal models was compared using the test datasets, as well as the remaining datasets that were not used for either training, validation or testing.

* Results for the Test Datasets
<table>
  <thead>
    <tr>
      <th align="center">Model</th>
      <th align="center">Number of Parameters</th>
      <th align="center">Overall Average MSE (in %)</th>
      <th align="center">Overall Average R2 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Vanilla RNN</td>
      <td align="center"><code>67,075</code></td>
      <td align="center"><code>0.0587</code></td>
      <td align="center"><code>99.9971</code></td>
    </tr>
    <tr>
      <td align="center">LSTM</td>
      <td align="center"><code>2,67,523</code></td>
      <td align="center"><code>0.0524</code></td>
      <td align="center"><code>99.9958</code></td>
    </tr>
    <tr>
      <td align="center">GRU</td>
      <td align="center"><code>2,01,475</code></td>
      <td align="center"><code>0.0136</code></td>
      <td align="center"><code>99.9992</code></td>
    </tr>
  </tbody>
</table>

* Results for the Miscellaneous Test Datasets
<table>
  <thead>
    <tr>
      <th align="center">Model</th>
      <th align="center">Number of Parameters</th>
      <th align="center">Overall Average MSE (in %)</th>
      <th align="center">Overall Average R2 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Vanilla RNN</td>
      <td align="center"><code>67,075</code></td>
      <td align="center"><code>0.1622</code></td>
      <td align="center"><code>99.9966</code></td>
    </tr>
    <tr>
      <td align="center">LSTM</td>
      <td align="center"><code>2,67,523</code></td>
      <td align="center"><code>0.1707</code></td>
      <td align="center"><code>99.9961</code></td>
    </tr>
    <tr>
      <td align="center">GRU</td>
      <td align="center"><code>2,01,475</code></td>
      <td align="center"><code>0.0418</code></td>
      <td align="center"><code>99.9991</code></td>
    </tr>
  </tbody>
</table>

## Conclusion
The GRU model achieved the lowest MSE and highest R2 score of all the models tested and hence, it proved to be quite capable of accurately estimating the SoC of a battery operating under different conditions.
