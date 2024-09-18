The following are instructions on how to run my code. It's mainly for training a pre-trained model but there's a streamlit application built as part of it that let's you go riffle through some images, their actual labels and the predicted. 

1. Clone this repo
```bash
git clone https://github.com/iampaapa/fellowship-ai-challenge.git
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

3. Install all the required dependencies using the requirements.txt file
```bash
pip install -r requirements.txt
```

4. Download the dataset and prep it for the training

    a. Download the dataset
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
```

b. Extract the dataset
```bash
tar -xzf 102flowers.tgz
```

c. Prepare the data by running the `data_prep.py` file
```bash
python data_prep.py
```

d. Remove unnecessary files
```bash
rm -rf jpg
rm 102flowers.tgz imagelabels.mat setid.mat
```

5. Train the pre-trained ResNet model
```bash
python train.py
```

6. Test out the web-app
```bash
streamlit run app.py
```