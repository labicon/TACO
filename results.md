- Scannet scene0000

    accuracy: 3.74/2.21
    
    completion: 3.41/2.64
    
    completion ratio: 87.67/95.33
    
    CD: 3.57/2.42
    
    precision@5cm: 84.80/95.26
    
    F1@5cm: 86.21/95.30


### Replica Room1 Evaluation

| Metric |Incremental-gamma1-inject100-frames_per_task100|Incremental-gamma1-inject1-frames_per_task1|Incremental-gamma1-inject1-frames_per_task1_trainskip20|Incremental-gamma1-inject1-frames_per_task1_trainskip20_iter100|Incremental-gamma1-inject1-frames_per_task1_trainskip20_iter1000| Incremental-gamma1-inject1-frames_per_task1_trainskip20_iter2000| Normal |Normal_trainskip20 |Normal_trainskip20_iter100 |
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


On Scannet scene0000

Co-SLAM*
accuracy: 2.22
completion: 2.58
completion ratio: 95.43
CD: 2.40
precision@5cm: 95.16
F1@5cm: 95.29

Co-SLAM(None)
accuracy: 29.73
completion: 27.29
completion ratio: 44.42
CD: 28.51
precision@5cm: 41.12
F1@5cm: 42.70

Ours(MAS loss + 100 + rho0.01 + mask1e-4)
accuracy: 2.73
completion: 2.91
completion ratio: 93.20
CD: 2.82
precision@5cm: 91.30
F1@5cm: 92.24

Ours(MAS loss + 10 + rho0.01 + mask1e-4 + gamma0.1)
accuracy: 3.15
completion: 2.78
completion ratio: 94.14
CD: 2.97
precision@5cm: 90.65
F1@5cm: 92.36

Ours(MAS loss + 100 + rho0.01 + mask1e-4 + gamma0.1)
accuracy: 3.84
completion: 3.21
completion ratio: 89.21
CD: 3.53
precision@5cm: 85.23
F1@5cm: 87.17

MAS_frames100
accuracy: 2.51
completion: 2.81
completion ratio: 94.33
CD: 2.66
precision@5cm: 92.76
F1@5cm: 93.54

MAS_frames10
accuracy: 2.39
completion: 2.68
completion ratio: 94.32
CD: 2.53
precision@5cm: 93.93
F1@5cm: 94.12

Ours after MAS loss 10
accuracy: 2.97
completion: 2.90
completion ratio: 92.94
CD: 2.93
precision@5cm: 90.73
F1@5cm: 91.82

Ours masked
accuracy: 2.57
completion: 2.91
completion ratio: 92.83
CD: 2.74
precision@5cm: 91.43
F1@5cm: 92.13

Ours after MAS loss 15
accuracy: 3.14
completion: 2.83
completion ratio: 93.40
CD: 2.98
precision@5cm: 91.37
F1@5cm: 92.37

Ours after MAS loss 20
accuracy: 3.19
completion: 2.80
completion ratio: 93.71
CD: 2.99
precision@5cm: 91.01
F1@5cm: 92.34

CADMM
accuracy: 7.24
completion: 3.97
completion ratio: 87.31
CD: 5.60
precision@5cm: 80.20
F1@5cm: 83.60

CNM
accuracy: 31.00
completion: 22.90
completion ratio: 27.44
CD: 26.95
precision@5cm: 19.97
F1@5cm: 23.11

accuracy: 32.74
completion: 15.06
completion ratio: 29.57
CD: 23.90
precision@5cm: 17.52
F1@5cm: 22.00






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

Ours (mas) 0.1
accuracy: 2.90
completion: 1.79
completion ratio: 94.97
CD: 2.34
precision@5cm: 90.31
F1@5cm: 92.58

Ours (masked) rho 0.5  gamma0.2
accuracy: 2.40
completion: 1.90
completion ratio: 94.29
CD: 2.15
precision@5cm: 91.53
F1@5cm: 92.89

Ours (masked) rho 0.2  gamma0.2
accuracy: 2.21
completion: 1.91
completion ratio: 94.41
CD: 2.06
precision@5cm: 92.72
F1@5cm: 93.56

Ours (masked) rho 0.1  gamma0.3
accuracy: 2.15
completion: 1.91
completion ratio: 94.47
CD: 2.03
precision@5cm: 92.97
F1@5cm: 93.71

Ours (masked) rho 0.1  gamma0.2
accuracy: 2.07
completion: 1.92
completion ratio: 94.37
CD: 2.00
precision@5cm: 93.29
F1@5cm: 93.83

Ours (masked) rho 0.1  gamma0.18
accuracy: 2.00
completion: 1.92
completion ratio: 94.39
CD: 1.96
precision@5cm: 93.76
F1@5cm: 94.08

Ours (masked) rho 0.1  gamma0.15
accuracy: 1.93
completion: 1.93
completion ratio: 94.33
CD: 1.93
precision@5cm: 94.12
F1@5cm: 94.23

Ours (masked) rho 0.1  gamma0.13
accuracy: 2.03
completion: 1.93
completion ratio: 94.47
CD: 1.98
precision@5cm: 93.60
F1@5cm: 94.03

Ours (masked) rho 0.1  gamma0.12
accuracy: 1.92
completion: 1.93
completion ratio: 94.52
CD: 1.92
precision@5cm: 94.30
F1@5cm: 94.41

Ours (masked) rho 0.1  gamma0.11
accuracy: 2.03
completion: 1.93
completion ratio: 94.47
CD: 1.98
precision@5cm: 93.60
F1@5cm: 94.03

Ours (masked) rho 0.1  gamma0.1
accuracy: 1.98
completion: 1.96
completion ratio: 94.38
CD: 1.97
precision@5cm: 94.00
F1@5cm: 94.19

Ours (masked) rho 0.1  gamma0.15
accuracy: 1.93
completion: 1.93
completion ratio: 94.33
CD: 1.93
precision@5cm: 94.12
F1@5cm: 94.23

Ours (masked) rho 0.08  gamma0.15
accuracy: 2.09
completion: 1.93
completion ratio: 94.45
CD: 2.01
precision@5cm: 93.19
F1@5cm: 93.82

Ours (masked) rho 0.12  gamma0.15
accuracy: 2.04
completion: 1.90
completion ratio: 94.56
CD: 1.97
precision@5cm: 93.46
F1@5cm: 94.01

Ours (masked) rho 1  gamma0.2
accuracy: 2.53
completion: 1.92
completion ratio: 94.07
CD: 2.22
precision@5cm: 90.85
F1@5cm: 92.43

Ours (masked) rho 1 gamma1
accuracy: 7.71
completion: 2.28
completion ratio: 92.71
CD: 5.00
precision@5cm: 85.19
F1@5cm: 88.79

Normal
accuracy: 1.79
completion: 1.74
completion ratio: 95.64
CD: 1.76
precision@5cm: 94.47
F1@5cm: 95.05

EWC
accuracy: 29.73
completion: 7.70
completion ratio: 65.20
CD: 18.72
precision@5cm: 32.25
F1@5cm: 43.16

MAS
accuracy: 2.91
completion: 1.79
completion ratio: 94.97
CD: 2.35
precision@5cm: 90.30
F1@5cm: 92.58

MAS_online
accuracy: 1.90
completion: 1.78
completion ratio: 95.41
CD: 1.84
precision@5cm: 93.94
F1@5cm: 94.67

Ours
accuracy: 1.88
completion: 1.94
completion ratio: 94.43
CD: 1.91
precision@5cm: 94.40
F1@5cm: 94.42

MAS 0.001
accuracy: 2.08
completion: 1.92
completion ratio: 94.46
CD: 2.00
precision@5cm: 93.26
F1@5cm: 93.85

UNIKD
accuracy: 5.22
completion: 2.04
completion ratio: 94.69
CD: 3.63
precision@5cm: 84.77
F1@5cm: 89.46

KR
accuracy: 8.97
completion: 2.16
completion ratio: 93.28
CD: 5.56
precision@5cm: 79.44
F1@5cm: 85.81

CNM(alpha12_weight0.1_20k)
accuracy: 6.29
completion: 2.17
completion ratio: 92.82
CD: 4.23
precision@5cm: 80.83
F1@5cm: 86.41

None
accuracy: 33.74
completion: 8.91
completion ratio: 60.31
CD: 21.32
precision@5cm: 30.26
F1@5cm: 40.30



Ours mask 1e-3
accuracy: 1.99
completion: 1.88
completion ratio: 94.79
CD: 1.94
precision@5cm: 93.50
F1@5cm: 94.14

Ours mask 1e-4
accuracy: 1.88
completion: 1.94
completion ratio: 94.43
CD: 1.91
precision@5cm: 94.40
F1@5cm: 94.42

Ours mask 1e-5
accuracy: 2.00
completion: 1.90
completion ratio: 94.60
CD: 1.95
precision@5cm: 93.96
F1@5cm: 94.28

Ours mask 1e-6
accuracy: 1.92
completion: 1.93
completion ratio: 94.52
CD: 1.92
precision@5cm: 94.30
F1@5cm: 94.41

Ours mask 1e-4 rho0.08
accuracy: 1.93
completion: 1.95
completion ratio: 94.56
CD: 1.94
precision@5cm: 94.13
F1@5cm: 94.34

Ours mask 1e-4 rho0.09
accuracy: 1.92
completion: 1.95
completion ratio: 94.50
CD: 1.93
precision@5cm: 94.21
F1@5cm: 94.36

Ours mask 1e-4 rho0.12
accuracy: 1.89
completion: 1.92
completion ratio: 94.59
CD: 1.91
precision@5cm: 94.28
F1@5cm: 94.44

Ours frames100
accuracy: 2.12
completion: 1.83
completion ratio: 95.03
CD: 1.98
precision@5cm: 92.76
F1@5cm: 93.88

MAS_frame1
accuracy: 2.11
completion: 1.79
completion ratio: 95.55
CD: 1.95
precision@5cm: 92.89
F1@5cm: 94.20

MAS_frame10
accuracy: 2.01
completion: 1.80
completion ratio: 95.34
CD: 1.91
precision@5cm: 93.22
F1@5cm: 94.27

MAS(100)
accuracy: 1.90
completion: 1.78
completion ratio: 95.41
CD: 1.84
precision@5cm: 93.94
F1@5cm: 94.67

Normal
accuracy: 1.79
completion: 1.74
completion ratio: 95.64
CD: 1.76
precision@5cm: 94.47
F1@5cm: 95.05

Ours new scale rho 0.01
accuracy: 1.85
completion: 1.78
completion ratio: 95.20
CD: 1.81
precision@5cm: 94.15
F1@5cm: 94.67

Ours new scale rho 0.01 frame 10
accuracy: 3.07
completion: 2.24
completion ratio: 94.69
CD: 2.65
precision@5cm: 88.32
F1@5cm: 91.40

Ours new scale rho 0.01 frame 10 gamma 0.1
accuracy: 2.38
completion: 1.98
completion ratio: 94.97
CD: 2.18
precision@5cm: 92.20
F1@5cm: 93.56

Ours new scale rho 0.01 frame 100 gamma 0.12
accuracy: 2.53
completion: 1.92
completion ratio: 94.92
CD: 2.23
precision@5cm: 90.46
F1@5cm: 92.64

Ours new scale rho 0.01 frame 100 gamma 0.1
accuracy: 2.31
completion: 1.92
completion ratio: 94.85
CD: 2.11
precision@5cm: 92.14
F1@5cm: 93.48

Ours new scale rho 0.01 frame 100 gamma 0.02
accuracy: 2.17
completion: 1.91
completion ratio: 94.79
CD: 2.04
precision@5cm: 92.63
F1@5cm: 93.70

Ours new scale rho 0.01 frame 100 gamma 0.06
accuracy: 2.27
completion: 1.87
completion ratio: 94.93
CD: 2.07
precision@5cm: 92.02
F1@5cm: 93.45