<h1 align="center"> Language Analytics exam portfolio in Cultural Data Science on Aarhus University</h1>
<h1 align="center"> Marie Damsgaard Mortensen</h1>
<h1 align="center"> May 2021</h1>

<p align="center">
  <a href="https://github.com/marmor97/cds-language-exam">
    <img src="examples/aarhus-university.png" alt="Logo" width="400" height="200">
  </a>

## :open_book: Content

This portfolio consists of 5 projects - 4 class assignments made throughout the semester (number 1-4) and 1 self-assigned (number 5). 


| Assignment | Description|
|--------|:-----------|
| 1 | Sentiment detection with SpaCy |
| 2 | Network analysis |
| 3 | Classifying sub-reddits of depression and suicide |
| 4 | Classifying seasons of Game of Thrones with Convolutional Neural Networks |
| 5 | Self-assigned project - Topic modelling of religious texts |
    

## :file_folder: Structure

To familiarize yourself with the structure of the repository, please see the table below that describes the main folders. Each assignment script, data and output is saved in the folder with the number corresponding to it: 
    
```bash

language-analytics-exam/  

├── src/  # Source scripts 
│   └── 1   
│   └── 2 
│   └── 3 
│   └── 4 
│   └── 5 
│    
├── data/  # Data 
│   └── 1   
│   └── 2 
│   └── 3 
│   └── 4 
│   └── 5 
│
├── out/  # Output files
│   └── 1
│   └── 2
│   └── 3
│   └── 4
│   └── 5
│
├── utils/  # Utility functions 
│   └── *.py
│
├── examples/ # Pictures etc. used in readme
│

```

## :wrench: Setup

To see and run the code with the correct packages installed, please clone the GitHub repository to a place on your computer where you'd like to have it by typing:

MAC / WORKER02

```
git clone https://github.com/marmor97/cds-language-exam # Clone repository to local machine or server

cd cds-language-exam # Change directory to the repository folder

bash create_lang_venv.sh # Creates a virtual environment
```

WINDOWS

```
git clone https://github.com/marmor97/cds-language-exam # Clone repository to local machine or server

cd cds-language-exam # Change directory to the repository folder

bash ./create_lang_venv_win.sh
```


Every time you wish to run any of the scripts, please type the following commands:

```
source lang101_marie/bin/activate # Activates virtual environment
```

Now you can move to any script in the src folder and execute it:

```
# Running script in assignment 1 as an example - exchange with whatever script you wish to run

cd src/{insert assignment number} # Changing to src and assignment folder 

python3 {insert script name} # Run script
    
```
To deactivate and remove the environment, the following commands need to be executed:

```
deactivate 

bash kill_lang_venv.sh

```
    
   
## :woman_technologist: License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## :question: Questions and contact  
For questions and other inquiries, please contact me on mariemortensen97@gmail.com.

