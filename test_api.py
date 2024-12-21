import requests
import json

def test_prediction():
    # Simplified test data with only relevant features
    test_data = {
        "service_plan": "Enterprise",
        "connection_type": "Fiber",
        "monthly_fee": 150.00,
        "data_usage_gb": 500.0,
        "avg_speed_mbps": 500.0,
        "uptime_percentage": 99.5,
        "payment_status": "Paid",
        "active_tickets": 1,
        "ticket_type": "Technical Issue",
        "ticket_status": "In Progress",
        "contract_start_date": "2023-01-15"
    }

    try:
        # Make prediction request
        response = requests.post(
            'http://localhost:5000/predict',
            json=test_data,
            headers={'Accept': 'application/json'}
        )
        
        # Ensure the request was successful
        response.raise_for_status()
        
        # Try to parse the response
        try:
            result = response.json()
            print("\nPrediction Response:")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print("\nError parsing prediction response:")
            print(f"Response text: {response.text}")
            print(f"Error: {str(e)}")
    except requests.exceptions.RequestException as e:
        print(f"\nError making prediction request: {str(e)}")

def test_training():
    try:
        # Make training request
        response = requests.post(
            'http://localhost:5000/train',
            headers={'Accept': 'application/json'}
        )
        
        # Ensure the request was successful
        response.raise_for_status()
        
        # Try to parse the response
        try:
            result = response.json()
            print("\nTraining Response:")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print("\nError parsing training response:")
            print(f"Response text: {response.text}")
            print(f"Error: {str(e)}")
    except requests.exceptions.RequestException as e:
        print(f"\nError making training request: {str(e)}")

if __name__ == '__main__':
    print("Testing API endpoints...")
    test_training()
    test_prediction() 