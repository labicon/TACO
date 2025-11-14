- Scannet scene0000

    accuracy: 3.74/2.21
    
    completion: 3.41/2.64
    
    completion ratio: 87.67/95.33
    
    CD: 3.57/2.42
    
    precision@5cm: 84.80/95.26
    
    F1@5cm: 86.21/95.30


### Replica Room1 Evaluation

| Metric                |Incremental-gamma1-inject100-frames_per_task100|Incremental-gamma1-inject1-frames_per_task1|Incremental-gamma1-inject1-frames_per_task1_trainskip20|Incremental-gamma1-inject1-frames_per_task1_trainskip20_iter100|Incremental-gamma1-inject1-frames_per_task1_trainskip20_iter1000| Incremental-gamma1-inject1-frames_per_task1_trainskip20_iter2000| Normal |Normal_trainskip20 |Normal_trainskip20_iter100 |
|------------------------|-------|-------|-------|-------|-------|-------|--------|--------|--------|
| **Accuracy**           | 7.60  | 4.94|5.52|8.22|9.00|10.59| 2.29|6.61|2.11|
| **Completion**         | 2.02  |2.02 |2.10|2.17|2.32|2.42|1.84 |1.99|1.85|
| **Completion Ratio**   | 93.37 |93.31|92.76|92.18|91.59|91.44|94.83|93.82|94.69|
| **Chamfer Distance (CD)** |4.81|3.48 |3.81|5.19|5.66|6.50| 2.07|4.30|1.98|
| **Precision @ 5 cm**   | 80.34 |87.09|84.70|79.58|81.41|81.99|92.02|83.95|92.81|
| **F1 @ 5 cm**          | 86.37 |90.09|88.55|85.42|86.20|86.46|93.40|88.61|93.74|





| **Metric** | **Incremental (inject100)** | **Incremental (inject1)** | **Normal** |
|-------------|-----------------------------|----------------------------|-------------|
| **Accuracy** | 7.60 | 4.94 | **2.29** |
| **Completion** | 2.02 | 2.02 | **1.84** |
| **Completion Ratio** | 93.37 | 93.31 | **94.83** |
| **Chamfer Distance (CD)** | 4.81 | 3.48 | **2.07** |
| **Precision @ 5 cm** | 80.34 | 87.09 |**92.02** |
| **F1 @ 5 cm** | 86.37 | 90.09 | **93.40** |

| **Metric** | **Incremental (inject1, trainskip=20)** | **Normal (trainskip=20)** |
|-------------|------------------------------------------|-----------------------------|
| **Accuracy** | **5.52** | 6.61 |
| **Completion** | 2.10 | **1.99** |
| **Completion Ratio** | 92.76 | **93.82** |
| **Chamfer Distance (CD)** | **3.81** | 4.30 |
| **Precision @ 5 cm** | **84.70** | 83.95 |
| **F1 @ 5 cm** | 88.55 | **88.61** |



| **Metric** | **iter=10** | **iter=100** | **iter=1000** | **iter=2000** | **Normal (iter=10)** | **Normal (iter=100)** |
|-------------|-------------|---------------|----------------|----------------|-----------------------------|------------------------|
| **Accuracy** | **5.52** | 8.22 | 9.00 | 10.59 |  6.61 |*2.11* |
| **Completion** | 2.10| 2.17 | 2.32 | 2.42 | **1.99** |*1.85* |
| **Completion Ratio** | 92.76 | 92.18 | 91.59 | 91.44 |**93.82** | *94.69* |
| **Chamfer Distance (CD)** | **3.81** | 5.19 | 5.66 | 6.50 |4.30 | *1.98* |
| **Precision @ 5 cm** | **84.70** | 79.58 | 81.41 | 81.99 |83.95 | *92.81* |
| **F1 @ 5 cm** | 88.55 | 85.42 | 86.20 | 86.46 |**88.61** | *93.74* |



room2
accuracy: 18.18
completion: 24.00
completion ratio: 41.16
CD: 21.09
precision@5cm: 47.74
F1@5cm: 44.20

office2
accuracy: 40.03
completion: 18.61
completion ratio: 33.01
CD: 29.32
precision@5cm: 40.50
F1@5cm: 36.38

room0
accuracy: 6.38
completion: 9.17
completion ratio: 64.92
CD: 7.77
precision@5cm: 61.10
F1@5cm: 62.95

office0
accuracy: 13.97
completion: 13.85
completion ratio: 58.34
CD: 13.91
precision@5cm: 50.32
F1@5cm: 54.03

office1
accuracy: 33.38
completion: 9.81
completion ratio: 50.87
CD: 21.60
precision@5cm: 37.04
F1@5cm: 42.87

office4
accuracy: 12.88
completion: 11.62
completion ratio: 52.32
CD: 12.25
precision@5cm: 41.41
F1@5cm: 46.23

office3
accuracy: 102.58
completion: 23.43
completion ratio: 35.64
CD: 63.00
precision@5cm: 24.16
F1@5cm: 28.80

room1
accuracy: 40.09
completion: 9.43
completion ratio: 58.05
CD: 24.76
precision@5cm: 28.24
F1@5cm: 37.99


Incre: 2.05it/s
Normal: 2.10it/s
Dropped  2.38%


Incre: 1760.56 MB
Normal: 3985.09 MB
Dropped 55.84%


scannet scene0000

Co-SLAM*
accuracy: 2.22
completion: 2.58
completion ratio: 95.43
CD: 2.40
precision@5cm: 95.16
F1@5cm: 95.29

Co-SLAM
accuracy: 29.73
completion: 27.29
completion ratio: 44.42
CD: 28.51
precision@5cm: 41.12
F1@5cm: 42.70

Ours
accuracy: 6.37
completion: 3.59
completion ratio: 88.78
CD: 4.98
precision@5cm: 83.14
F1@5cm: 85.87

CADMM
accuracy: 7.24
completion: 3.97
completion ratio: 87.31
CD: 5.60
precision@5cm: 80.20
F1@5cm: 83.60


room1 

gamma0.01
accuracy: 6.91
completion: 2.02
completion ratio: 93.36
CD: 4.46
precision@5cm: 83.96
F1@5cm: 88.41

gamma0.05
accuracy: 5.60
completion: 2.00
completion ratio: 93.44
CD: 3.80
precision@5cm: 85.53
F1@5cm: 89.31

gamma0.2
accuracy: 5.11
completion: 2.01
completion ratio: 93.34
CD: 3.56
precision@5cm: 85.87
F1@5cm: 89.45

gamma0.4
accuracy: 5.25
completion: 2.05
completion ratio: 93.01
CD: 3.65
precision@5cm: 85.44
F1@5cm: 89.07

gamma0.6
accuracy: 5.22
completion: 2.06
completion ratio: 92.95
CD: 3.64
precision@5cm: 85.52
F1@5cm: 89.08

gamma0.8
accuracy: 5.74
completion: 2.05
completion ratio: 93.07
CD: 3.90
precision@5cm: 84.75
F1@5cm: 88.71

gamma1.2
accuracy: 5.86
completion: 2.11
completion ratio: 92.75
CD: 3.98
precision@5cm: 84.28
F1@5cm: 88.31

Normal
accuracy: 6.66
completion: 1.99
completion ratio: 93.91
CD: 4.32
precision@5cm: 83.82
F1@5cm: 88.58




On room1
CADMM
accuracy: 4.65
completion: 1.95
completion ratio: 93.78
CD: 3.30
precision@5cm: 87.32
F1@5cm: 90.43

gamma0.2
accuracy: 3.15
completion: 1.93
completion ratio: 94.11
CD: 2.54
precision@5cm: 90.29
F1@5cm: 92.16

gamma1
accuracy: 3.70
completion: 1.90
completion ratio: 94.04
CD: 2.80
precision@5cm: 89.29
F1@5cm: 91.60

Normal
accuracy: 1.79
completion: 1.74
completion ratio: 95.64
CD: 1.76
precision@5cm: 94.47
F1@5cm: 95.05

