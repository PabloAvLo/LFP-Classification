# LFP Classification
Classifier of Local Field Potential signals using Deep Neural Networks for head angular direction prediction 

## Prerequisites
The current project was developed using **Ubuntu 18.04.4 LTS**.
The following are the required packages and python libraries.
It is recommended to use the listed versions, nevertheless is worth to try the newest available version of them.

### Packages:
```
1. Python (3.6) 
```

**Optional:**
```
1. Latex
2. Doxygen
3. Make
4. Git
```

### Python libraries:
Package | Version
------- | --------
pip3 | 20.1
numpy | 1.18.4
matplotlib | 3.2.1 
tensorflow | 2.2.0
keras | 2.3.1
sklearn | 0.0
pandas | 1.0.3
scikit-learn | 0.23.0
seaborn | 0.10.1


## Install

Check the installed Linux version using: `lsb_release -a`.
Check the installed version of the package using: `<package> --<version>`
Check the installed python libraries and their version using: `pip list`

### Packages:
```
sudo apt-get install python3.6 texlive-latex-extra doxygen doxygen-gui make git
```

### Pip:

This is necessary to install the rest of python libraries

```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.6 get-pip.py
```

### Python libraries:

If getting the latest available version:
```
pip3 install --upgrade <package>
```

To download an specific version:
```
pip3 install <package>==<version>
```
 
## Usage


## Author
**Pablo Avila** - [PabloAvLo](https://github.com/PabloAvLo)

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* The present code is part of my grade thesis in Electrical Engineering at the University of Costa Rica.

* The main objective is to prove the hypothesis that based on Local Field Potential measurments from the Entorhinal Cortex of the Hippocampus of rats is possible to extract angular information of the head direction of the subject. 

* The dataset used in this experiment is: [BuszakiLab HC-3 Dataset](https://crcns.org/data-sets/hc/hc-3/about-hc-3/?searchterm=LFP).

* In order to do so, machine learning algorithms will be applied to the data, such as Feedforward Neural Network, RNNs and CNNs.
