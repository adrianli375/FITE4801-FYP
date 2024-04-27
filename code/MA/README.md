# Moving Average Strategy

The moving average (MA) strategy, with the deep learning implementation to predict future volatility, is our final algorithm. This developed algorithm is one of our major deliverables. 

Different versions are developed which leads to the final algorithm. The table below shows the characteristics for each improvement of the algorithm. 

|  Version  |                                        Enhancements                                       |                              Remarks                             |
|:---------:|:-----------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| v0        | N/A                                                                                       | The baseline moving average strategy.                            |
| v1        | Added a pair of MA bands to open trades                                                   |                                                                  |
| v2        | Added a pair of MA bands to open and close trades                                         |                                                                  |
| v3      | Volatility adjustment to the MA bands, based on v2                                        |                                                                        |
| v4        | Use two pairs of MA bands to open and close trades                                        |                                                                  |
| v5        | Adopted a dynamic moving average                                                          |                                                                  |
| v6        | Added the LSTM model to predict future volatility                                         |                                                                  |
| v7 (dev)  | Added adjustment to previous losses, based on v5                                          | For development purpose                                                                 |
| v7 (prod) | Added adjustment to previous losses, based on v6                                          | Without Retrain                                                                 |
| v7.1      | Added adjustment to previous losses, with retrain mechanism of the LSTM model             | The actual "v7" used in the cryptocurrency market                |
| v7.2      | Added adjustment to previous losses, with progressive retrain mechanism of the LSTM model | The actual "v7" used in the US stock market                      |
