import pytest
from app import app, db

# Define valid input for prediction test
valid_input = {
    "bedrooms": 2,
    "bathrooms": 1,
    "accommodates": 4,
    "neighbourhood_cleansed": "South Boston"
}

# Define invalid neighborhood input for prediction test
invalid_neighbourhood_input = {
    "bedrooms": 2,
    "bathrooms": 1,
    "accommodates": 4,
    "neighbourhood_cleansed": "Invalid Neighborhood"
}

# Define missing field input for prediction test
missing_field_input = {
    "bedrooms": 2,
    "bathrooms": 1,
    "neighbourhood_cleansed": "South Boston"
}


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client


def test_reload_data(client):
    """Test the reload endpoint that loads the data."""
    response = client.post('/reload')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'total_listings' in json_data
    assert 'average_price' in json_data

def test_predict_after_reload(client):
    """Test prediction endpoint after reloading the data."""
    # Reload the data first
    client.post('/reload')

    # Test valid prediction
    response = client.post('/predict', json=valid_input)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'predicted_price' in json_data


def test_invalid_neighbourhood(client):
    """Test prediction with an invalid neighbourhood."""
    # Reload the data first
    client.post('/reload')

    # Test invalid neighborhood
    response = client.post('/predict', json=invalid_neighbourhood_input)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid neighborhood" in json_data['error']


def test_missing_fields(client):
    """Test prediction with missing fields."""
    # Reload the data first
    client.post('/reload')

    # Test with missing fields
    response = client.post('/predict', json=missing_field_input)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid numeric values for bedrooms, bathrooms, or accommodates" in json_data['error']

