# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import shotchartdetail
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Step 1: Fetch shot data from NBA Stats API
def fetch_nba_shot_data(team_id, season='2022-23'):
    """
    access shot chart data for a specific team/season using NBA Stats API
    """
    shot_chart = shotchartdetail.ShotChartDetail(
        team_id=team_id,  
        player_id=0,  # 0 for team-level data
        season_nullable=season,
        season_type_all_star='Regular Season',
        context_measure_simple='FGA'  # Field Goal Attempts
    )
    shot_data = shot_chart.get_data_frames()[0]
    return shot_data

# Step 2: Data Preprocessing
def preprocess_data(shot_data):
    """
    clean and process shot data for analysis
    """
    # Select relevant columns
    columns = ['PLAYER_NAME', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_DISTANCE', 
               'ACTION_TYPE', 'SHOT_MADE_FLAG', 'PERIOD', 'SHOT_ATTEMPTED_FLAG', 
               'LOC_X', 'LOC_Y']
    shot_data = shot_data[columns]
    
    # Remove rows with missing values
    shot_data = shot_data.dropna()
    
    # Encode categorical variables
    shot_data['SHOT_TYPE'] = shot_data['SHOT_TYPE'].map({'2PT Field Goal': 2, '3PT Field Goal': 3})
    shot_data['SHOT_ZONE_BASIC'] = shot_data['SHOT_ZONE_BASIC'].astype('category').cat.codes
    shot_data['ACTION_TYPE'] = shot_data['ACTION_TYPE'].astype('category').cat.codes
    
    return shot_data

# Step 3: EDA
def perform_eda(shot_data, output_dir='eda_plots'):
    """
    perform EDA & save visualizations 
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Shot success rate by zone
    zone_success = shot_data.groupby('SHOT_ZONE_BASIC')['SHOT_MADE_FLAG'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=zone_success.index, y=zone_success.values)
    plt.title('Shot Success Rate by Zone')
    plt.xlabel('Shot Zone (Encoded)')
    plt.ylabel('Success Rate')
    plt.savefig(f'{output_dir}/shot_success_zone.png')
    plt.close()
    
    # Shot distance vs. success
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='SHOT_DISTANCE', y='SHOT_MADE_FLAG', data=shot_data, alpha=0.3)
    plt.title('Shot Distance vs. Success')
    plt.xlabel('Shot Distance (feet)')
    plt.ylabel('Shot Made (1 = Yes, 0 = No)')
    plt.savefig(f'{output_dir}/shot_distance_success.png')
    plt.close()
    
    # Shot locations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='LOC_X', y='LOC_Y', hue='SHOT_MADE_FLAG', data=shot_data, alpha=0.3)
    plt.title('Shot Locations (Made vs. Missed)')
    plt.xlabel('Court X (feet)')
    plt.ylabel('Court Y (feet)')
    plt.savefig(f'{output_dir}/shot_locations.png')
    plt.close()

# Step 4: Build Classification Model
def train_shot_model(shot_data):
    """
    train a RF model to predict shot success
    """
    features = ['SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_DISTANCE', 'ACTION_TYPE', 'PERIOD', 'LOC_X', 'LOC_Y']
    X = shot_data[features]
    y = shot_data['SHOT_MADE_FLAG']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RF
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Predict probabilities for all shots
    shot_data['SHOT_PROB'] = model.predict_proba(X)[:, 1]
    
    return shot_data, model

# Step 5: A/B Testing Simulation
def perform_ab_testing(shot_data):
    """
    simulate A/B testing to compare actual vs. model-recommended shots.
    """
    # Define high-probability threshold for recommended shots
    shot_data['RECOMMENDED'] = shot_data['SHOT_PROB'] > 0.5
    
    # Simulate: Group A (actual shots), Group B (recommended shots)
    group_a = shot_data[shot_data['SHOT_ATTEMPTED_FLAG'] == 1]['SHOT_MADE_FLAG']
    group_b = shot_data[shot_data['RECOMMENDED'] == 1]['SHOT_MADE_FLAG']
    
    # Perform t-test to compare success rates
    t_stat, p_value = ttest_ind(group_a, group_b, equal_var=False)
    print(f"A/B Testing Results:\nT-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    
    # Expected points calculation
    shot_data['EXPECTED_POINTS'] = shot_data['SHOT_PROB'] * shot_data['SHOT_TYPE']
    actual_points = shot_data[shot_data['SHOT_ATTEMPTED_FLAG'] == 1]['SHOT_MADE_FLAG'] * shot_data[shot_data['SHOT_ATTEMPTED_FLAG'] == 1]['SHOT_TYPE']
    recommended_points = shot_data[shot_data['RECOMMENDED'] == 1]['EXPECTED_POINTS']
    
    print(f"Actual Points (Mean per shot): {actual_points.mean():.4f}")
    print(f"Recommended Points (Mean per shot): {recommended_points.mean():.4f}")

# Main execution
if __name__ == "__main__":
    # Fetch data for LA Lakers (team ID: 1610612747)
    shot_data = fetch_nba_shot_data(team_id=1610612747, season='2022-23')
    
    # Preprocess data
    shot_data = preprocess_data(shot_data)
    
    # Save to CSV for Shiny
    shot_data.to_csv('shot_data.csv', index=False)
    
    # Perform EDA
    perform_eda(shot_data)
    
    # Train model and get predictions
    shot_data, model = train_shot_model(shot_data)
    
    # Perform A/B testing
    perform_ab_testing(shot_data)
    
    # Save final dataset with predictions
    shot_data.to_csv('enhanced_shot_data.csv', index=False)
    print("Data saved to 'enhanced_shot_data.csv' for Shiny dashboard.")
