## Expected Runs Model for Cricket Ball-Tracking Data
# Overview
This project involves building an expected runs model using ball-tracking data for cricket. The dataset contains over 200,000 balls, each with ball-tracking attributes (e.g., speed, swing angle, bounce position) and outcomes (runs and wickets). The goal is to predict the expected runs for each ball using machine learning techniques.

# The project is structured as an end-to-end Data Science workflow, including:

Exploratory Data Analysis (EDA)

Feature Engineering

Model Selection and Training

Model Evaluation

Answering Task Questions

# Dataset
The dataset contains the following columns:

Column Name	Description
release_speed_kph	Speed of the ball in KPH at release point
swing_angle	Amount of swing in the air (degrees)
deviation	Amount of movement off the pitch (degrees)
release_position_y	Bowler release position (y-coordinate)
release_position_z	Bowler release position (z-coordinate)
bounce_position_y	Ball bounce position (y-coordinate)
bounce_position_x	Ball bounce position (x-coordinate)
crease_position_y	Ball position at crease level (y-coordinate)
crease_position_z	Ball position at crease level (z-coordinate)
stumps_position_y	Ball position at stumps level (y-coordinate)
stumps_position_z	Ball position at stumps level (z-coordinate)
bounce_velocity_ratio_z	Ratio of ball velocity after and before bounce
release_angle	Angle of the ball at release
drop_angle	Angle of the ball just before bounce
bounce_angle	Angle of the ball just after bounce
batting_hand	Batsman handedness (right/left)
bowling_hand	Bowler handedness (right/left)
bowling_type	Bowler type (pace/spin)
runs	Runs scored off the bat (excluding extras)
wicket	1 if a wicket is attributed to the bowler (e.g., caught, bowled), else 0
#Steps
1. Exploratory Data Analysis (EDA)
Checked for missing values and outliers.

Analyzed the distribution of the target variable (runs).

Visualized correlations between features using a heatmap.

2. Feature Engineering
Encoded categorical variables (batting_hand, bowling_hand, bowling_type) using one-hot encoding.

Scaled numerical features using StandardScaler.

3. Model Selection and Training
Experimented with three models:

#Linear Regression

Random Forest Regressor

XGBoost Regressor

# Split the data into training and testing sets (80-20 split).

4. Model Evaluation
Evaluated models using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R-squared (R²)

Visualized actual vs. predicted runs and residual plots.

5. Answering Task Questions
Provided detailed answers to the task questions based on the model's performance and insights.

How to Run the Code
Prerequisites
Python 3.x

# Google Colab or Jupyter Notebook

Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

Steps
Open Google Colab:

Go to Google Colab.

Create a new notebook.

Upload the Dataset:

If the dataset is on your local machine, upload it to Colab using:

python
Copy
from google.colab import files
uploaded = files.upload()
If the dataset is in Google Drive, mount Google Drive:

python
Copy
from google.colab import drive
drive.mount('/content/drive')
Copy and Paste the Code:

Copy the code from the project and paste it into the Colab notebook.

Update the file path in the pd.read_csv() function to point to your dataset.

Run the Code:

Execute each cell step by step.

# Results
Model Performance
XGBoost performed the best among the three models, with the lowest MAE and highest R² score.

# Key Insights
Deliveries with specific swing angles, bounce positions, and release speeds are harder to score off.

The model's predictions closely match the variance of the target distribution (runs).

#Future Steps
Perform advanced feature engineering (e.g., interaction features).

Experiment with hyperparameter tuning and ensemble methods.

Explore deep learning models (e.g., Neural Networks) if computational resources allow.

# Questions and Answers
1. Which Machine Learning model have you picked and why?
I picked XGBoost because it provides the best balance of accuracy and performance, capturing complex relationships in the data.

2. How are you evaluating the results of your model?
I am using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) to evaluate the model. Additionally, I am visualizing actual vs. predicted runs and residual plots.

3. Does the variance of your model's output distribution match the variance of the target distribution (runs)?
The variance of the model's output distribution should ideally match the target distribution. If not, it indicates underfitting or overfitting. XGBoost typically matches the variance better than simpler models.

4. Explaining the model's learnings to a coach
The model suggests that deliveries with specific swing angles, bounce positions, and release speeds are harder to score off. Coaches can use this to strategize bowling plans.

5. Future steps
Future steps include advanced feature engineering, hyperparameter tuning, and experimenting with ensemble methods or neural networks.

# License
This project is proprietary and confidential. The dataset and code are for recruitment purposes only and must not be shared or distributed.
