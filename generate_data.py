from faker import Faker
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class ISPDataGenerator:
    def __init__(self):
        self.fake = Faker()
        self.current_date = datetime.now()
        
        # Define service plans and their characteristics
        self.service_plans = {
            'Basic': {'speed_range': (50, 100), 'fee_range': (30, 50)},
            'Standard': {'speed_range': (100, 300), 'fee_range': (50, 80)},
            'Premium': {'speed_range': (300, 500), 'fee_range': (80, 120)},
            'Enterprise': {'speed_range': (500, 1000), 'fee_range': (120, 200)}
        }
        
        # Define connection types and their reliability
        self.connection_types = {
            'Fiber': {'uptime_range': (99.0, 99.9), 'speed_factor': 1.0},
            'Cable': {'uptime_range': (97.0, 99.5), 'speed_factor': 0.9},
            'DSL': {'uptime_range': (95.0, 98.5), 'speed_factor': 0.8},
            'Satellite': {'uptime_range': (94.0, 97.5), 'speed_factor': 0.7}
        }
        
        self.ticket_types = ['Technical Issue', 'Billing Query', 'Service Request', 'Connection Problem']
        self.ticket_statuses = ['Open', 'In Progress', 'Resolved', 'Closed']
        self.payment_statuses = ['Paid', 'Pending', 'Overdue']

    def generate_customer_id(self, index):
        return f'CUST{str(index).zfill(6)}'

    def generate_dates(self):
        # Contract start date between 2 years ago and now
        contract_start = self.fake.date_between(start_date='-2y', end_date='now')
        
        # Last payment date between contract start and now
        last_payment = self.fake.date_between(start_date=contract_start, end_date='now')
        
        # Last ticket date between contract start and now
        last_ticket = self.fake.date_between(start_date=contract_start, end_date='now')
        
        return contract_start, last_payment, last_ticket

    def calculate_satisfaction(self, row):
        """Calculate customer satisfaction based on service metrics"""
        base_satisfaction = 7.0  # Base satisfaction score
        
        # Speed satisfaction (weight: 0.3)
        speed_ratio = row['avg_speed_mbps'] / row['promised_speed']
        speed_satisfaction = min(1.0, speed_ratio) * 3
        
        # Uptime satisfaction (weight: 0.2)
        uptime_satisfaction = (row['uptime_percentage'] - 94) / (100 - 94) * 2
        
        # Price satisfaction (weight: 0.2)
        price_per_mbps = row['monthly_fee'] / row['avg_speed_mbps']
        price_satisfaction = (1 - min(1, price_per_mbps / 0.5)) * 2
        
        # Support satisfaction (weight: 0.3)
        ticket_factor = max(0, 1 - (row['active_tickets'] * 0.5))
        support_satisfaction = ticket_factor * 3
        
        # Calculate total satisfaction
        satisfaction = base_satisfaction + speed_satisfaction + uptime_satisfaction + price_satisfaction + support_satisfaction
        
        # Add some random variation (-0.5 to 0.5)
        satisfaction += random.uniform(-0.5, 0.5)
        
        # Clip to range 1-10
        return max(1, min(10, satisfaction))

    def generate_data(self, num_records):
        data = []
        
        for i in range(num_records):
            # Basic customer info
            service_plan = random.choice(list(self.service_plans.keys()))
            connection_type = random.choice(list(self.connection_types.keys()))
            plan_specs = self.service_plans[service_plan]
            conn_specs = self.connection_types[connection_type]
            
            # Generate dates
            contract_start, last_payment, last_ticket = self.generate_dates()
            
            # Calculate service metrics
            promised_speed = random.uniform(*plan_specs['speed_range'])
            actual_speed = promised_speed * conn_specs['speed_factor'] * random.uniform(0.9, 1.1)
            monthly_fee = random.uniform(*plan_specs['fee_range'])
            uptime = random.uniform(*conn_specs['uptime_range'])
            
            # Generate row data
            row = {
                'customer_id': self.generate_customer_id(i + 1),
                'full_name': self.fake.name(),
                'email': self.fake.email(),
                'phone': self.fake.phone_number(),
                'address': self.fake.address(),
                'service_plan': service_plan,
                'connection_type': connection_type,
                'monthly_fee': round(monthly_fee, 2),
                'data_usage_gb': round(random.uniform(50, 1000), 2),
                'avg_speed_mbps': round(actual_speed, 2),
                'promised_speed': round(promised_speed, 2),
                'uptime_percentage': round(uptime, 2),
                'payment_status': random.choices(
                    self.payment_statuses, 
                    weights=[0.8, 0.15, 0.05]
                )[0],
                'last_payment_date': last_payment.strftime('%Y-%m-%d'),
                'contract_start_date': contract_start.strftime('%Y-%m-%d'),
                'active_tickets': random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.08, 0.02])[0],
                'ticket_type': random.choice(self.ticket_types),
                'ticket_status': random.choice(self.ticket_statuses),
                'last_ticket_date': last_ticket.strftime('%Y-%m-%d')
            }
            
            # Calculate customer satisfaction
            row['customer_satisfaction'] = round(self.calculate_satisfaction(row), 2)
            
            data.append(row)
        
        return pd.DataFrame(data)

def main():
    # Initialize generator
    generator = ISPDataGenerator()
    
    # Generate 5000 records
    print("Generating ISP customer data...")
    df = generator.generate_data(5000)
    
    # Save to CSV
    output_file = 'isp_admin_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nData generated and saved to {output_file}")
    
    # Print data summary
    print("\nDataset Summary:")
    print(f"Total records: {len(df)}")
    print("\nSatisfaction Score Distribution:")
    print(df['customer_satisfaction'].describe())
    
    print("\nCorrelations with Customer Satisfaction:")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlations = df[numeric_cols].corr()['customer_satisfaction'].sort_values(ascending=False)
    print(correlations)

if __name__ == "__main__":
    main() 