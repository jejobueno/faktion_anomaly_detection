## Faktion Anomaly Detection

This use-case project is provided Faktion. In this project, we have to devlop an AI model to detect if a dice is normal or has some imperfections.

**Success criterias:**  
*...*

----------
### Project Guidelines
- Repository: `faktion-anomaly-detection`
- Type of Challenge: `Use-case`
- Duration: `2 weeks`
- Deadline: `14/10/2021`
- Team Challenge : `Team (3 people)`
-------
### Data

The dataset provided for this project contains a set of images with normal dices, and a set with anormal dices
<img alt="Normal Dice" src="assets/normal_dice_2.jpg" />
<img alt="Anormal Dice" src="assets/anomalous_dice_2.jpg" />

--------
### Prerequisites

- **Python3.8+**

### Installation

```bash
git clone git@github.com:kaygu/faktion_anomaly_detection.git
cd faktion_anomaly_detection
pip install -r requirements.txt 
```
### Usage

```bash
python main.py
```
----------
### How it works
1. Upload your image.
2. During the upload it will use Neural network (CNN) to classify the dice.
3. Then for the second part it use OpenCV to detect anomaly and confirm if it's a broken dice or not.
### Contributors

- [Camille De Neef](https://github.com/kaygu)
- [Jesus Bueno](https://github.com/jejobueno) 
- [Kadri Salija](https://github.com/misterkadrix)
