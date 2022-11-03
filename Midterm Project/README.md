# Taxi Fare Prediction, a MLZoomcamp Project

## 1. Prelude
this repository and project are a part of the amazing [Machine Learning Bootcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp).
Although this dataset is not unique any sort of way, I have worked with it often in the little time that I have spent in the data field.

It is no secret the greater part of a data science project is the preprocessing, wherein domain knowledge plays an indespensible role, sadly my fields of expertise are not well available publicly, so I used something that I am at the very least familiar with.

### 1.1 Requirements

#### 1.1.2 The Virtual Environment
Poetry was used as the virtual environment tool while developing this project. this does not include much, just sklearn and xgboost mainly.
Poetry offers a lot of benefits and [potentially better performance](https://plainenglish.io/blog/poetry-a-better-version-of-python-pipenv)!
to get started it is as simple as:
```python
# 1. install poetry
pip install poetry

#2. initialize it and define dependencies (if you do not have a pyproject.toml file, otherwise you can just skip to next step)
poetry init

#3 install said dependencies
poetry install

# bonus: to add a dependency simply use:
poetry add <dependency>

#4 to use the environment
poetry shell
```
**NOTE** the model and prediction service are containerized via BentoML, and all dependencies were stated in the Bentofile.yaml, which takes care of all dependences. so to run this project, you do not need to install the poetry environment, it is just mentioned for completeness. to read more you can check the [docs](https://python-poetry.org/docs/), they are pretty comprehensive, but unlike usual docs, they are not full of bloat, I really enjoy skimming through them.

#### 1.1.3 CLI Tools
If you want to run the project, you will need [gcloud sdk](https://cloud.google.com/sdk/docs/install) installed, as some commands will be run through them. 


## 2. Overview

### 2.1 Probelm Statement:
A for hire vehicles/taxi company wants to launch a service to predict the fare. basically just like the regular Uber app,  it gives you an estimate of the ride before you order the ride itself.
How does it do it? well, there are many things at play, and of course a real life implementation would be vastly different than this prototype, but it is a starting point.
There is a tremendous amount of data from rides in the past, which can be used to infer the approximate cost of a ride from location X to location Y given different circumstances, however to get to the point where you are making reasonable predictions, some work is needed, this is what we will explore together.


### 2.2 The Data:
#### 2.2.1 Main Data Source
the so very famous NY Taxi dataset, if you are in the field, you must have come across it. 
Kindly note that there are 2 versions circling around the internet; [the raw version](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and the [kaggle competition version](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction). 
The difference is not severe, the kaggle version is basically prepared for machine learning in a way, no catgorical variables or such. It is the easier dataset to take for a quick ride  – pun intended – but in this project we will start out with the raw version.

the link is provided in the notebook, however in case you do not want to run the notebook, you can get the used file from here
```bash
!wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-06.parquet
```
The description for each of the columns in the dataset can be found [here](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).
#### 2.2.2 Extra Data for Extra Features
The kaggle dataset comes with longitude and latitude coordinates of the pickup and dropoff locations, the raw version does not. this is a very important feature, not intrinsicly in itself, but because we can use them to extract other valuable features, like pure distance between 2 points for example. 
To do this, we will need a shapefile. I have included the final CSV, however if you are interested you can get the shape file from [here](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip) and use [this](https://disq.us/url?url=https%3A%2F%2Fmygeodata.cloud%2Fconverter%2Fshp-to-latlong%29%3APxLSP7AjU7-LLKjz3b1k2lZO5fc&cuid=5002428) to convert it to a usable form with proper coordinates.
After that, with some functions, the shape file was turned to a dataframe, which was merged with the main data to provide the longitude and latitude needed.

#### 2.2.3 Honorable mention
shoutout to OpenStreetMap, with help of their API, I was able to improve this model. 
To elaborate, as you will see, trip duration plays a reasonable role in the fare prediction. well, you can get the trip duration from the data by subtracting dropoff and pickup times, but you will not ask a user to tell you how long the trip will last right? with the API, we can provide coordinates and it will tell us the duration.
The logic can be found in the notebook but here you go for a quick sneapeak.

```python
    def get_trip_duration(self, df):
        r = requests.get(f"http://router.project-osrm.org/route/v1/car/{df.pickup_long[0]},{df.pickup_lat[0]};{df.dropoff_long[0]},{df.dropoff_lat[0]}?overview=false""")
        routes = json.loads(r.content)
        trip_duration = routes.get("routes")[0]['duration']
        trip_duration = trip_duration/60
        return trip_duration
```

## 3. EDA 

You can see the results of the EDA in the first part of the notebook, ranges of values, missing values, analysis of target variable and feature importance analysis were carried out. 
a small - and definitly not encompassing - summary:
- there are a good number of null values
- there are data quality issues
- the data is for July 2022, but there are 3 other years in the dataset.
- there are 42 different unique days
- a lot of fare/money related metrics have negative values
- the distribution of the target is skewed
- there are trips that have covered obsence distances or accrued obsence amounts of "fare_amount"
- most if not all categorical variables that start with the dataset provide no benefit to the model's performance
- the target variable seems to be affected most by time and duration of the trip, which makes perfect sense.

## 4. Model training	

In the second half of the notebook I started with base modelling.
the following models were sampled:
1. Linear Regression
2. RandomForestRegressor
3. GradientBoostRegressor
4. LGBM
5. XGBOOST

XGBOOST was winning, with GradientBoostingRegressor being slightly behind.
Gridsearch and hyperparameter tuning was carried out, LGBM, GradientBoostRegressor as well as XGBOOST were performing very closely, but after some trials, I felt like XGBOOST is more stable and has lower variance, i.e it performs better across different datasets, so that was the choosen model.

## 5. Exporting notebook to script	
the main functions and logic in the notebook has been exported to train.py, it have a main function that takes your data as input to run

## 6. Reproducibility
You should be able to run the notebook error free, the data is commited to the repository or you are guided to download it.
If you provide data to the training script, you should be able to run it

## 7. Model deployment	
Model is containerized, packaged, deployed and served via BentoML

## 8.Dependency and enviroment management	
Bentofile.yaml

## 9. Containerization
The app was containerized via BentoML, which takes the Dockerfile creation part out of the game. the BentoML Dockerfile is presented anyhow.
Were it not for BentoML, I would have used a dockerfile similar to this:

```Dockerfile
FROM python:3.9-slim as base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y gcc libffi-dev g++
WORKDIR /app

FROM base as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.1.3

RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

COPY pyproject.toml poetry.lock ./
RUN . /venv/bin/activate && poetry install --no-dev --no-root

COPY . .
RUN . /venv/bin/activate && poetry build

FROM base as final

COPY --from=builder /venv /venv
COPY --from=builder /app/dist .
COPY docker-entrypoint.sh ./

RUN . /venv/bin/activate && pip install *.whl
CMD ["./docker-entrypoint.sh"]

```

this is called multistage build, basically you try to create a final image that is as light as possible, by installing dependencies that are only needed during the build in an early stage, effectively leaving them behind in the final image. with the docker-entrypoint.sh you could start your flask/gunicorn server. 

## 10. Cloud Deployment

The service was deployed to GCP Cloud Run. 

You can do it too in under 5 minutes if you've used BentoML.

here is how:

```bash
# authenticate 
gcloud auth configure-docker

# get model location
saved_path=$(bentoml get ny_taxi_fare_predictor:cpix4mk3qsbtrbfj -o path --quiet)

# assuming you have containerized your app via BentoML bentoml containerize app:tag, the image would alreayd be created (check by using docker images)
docker tag ny_taxi_fare_predictor:cpix4mk3qsbtrbfj gcr.io/${project_id}/image_repo/ny_taxi_fare_predictor:cpix4mk3qsbtrbfj

# push it to container registry
docker push gcr.io/${project_id}/image_repo/ny_taxi_fare_predictor:cpix4mk3qsbtrbfj

# deploy it with cloud run
gcloud run deploy nytaxi-prediction --image gcr.io/${project_id}//ny_taxi_fare_predictor:cpix4mk3qsbtrbfj --allow-unauthenticated --port 3000
```
and viola, you are done. here is a video that showcases this.

[![Everything Is AWESOME](https://yt-embed.herokuapp.com/embed?v=jFcxnZEAoX4)](https://www.youtube.com/embed/jFcxnZEAoX4 "Everything Is AWESOME")


Link: https://nytaxi-prediction-q2litgafla-ew.a.run.app/#/Service%20APIs
but cannot guarantee it will stay functional for long.
to test it you need to enter 3 things
1. passenger count
2. pickup zone
3. dropoff zone

and it should respond back with the prediction.
