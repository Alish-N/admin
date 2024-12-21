from model_preprocessing import ISPCustomerSatisfaction

def format_metrics(metrics):
    """Format metrics for display"""
    print("\nModel Performance Metrics:")
    
    print("\nTraining Metrics:")
    print("Regression Metrics:")
    print(f"R² Score: {metrics['train_metrics']['regression']['r2']:.4f}")
    print(f"Adjusted R²: {metrics['train_metrics']['regression']['adjusted_r2']:.4f}")
    print(f"Explained Variance: {metrics['train_metrics']['regression']['explained_variance']:.4f}")
    print(f"MAE: {metrics['train_metrics']['regression']['mae']:.4f}")
    print(f"RMSE: {metrics['train_metrics']['regression']['rmse']:.4f}")
    
    print("\nClassification Metrics (Threshold = 7.0):")
    print(f"Accuracy: {metrics['train_metrics']['classification']['accuracy']:.4f}")
    print(f"Precision: {metrics['train_metrics']['classification']['precision']:.4f}")
    print(f"Recall: {metrics['train_metrics']['classification']['recall']:.4f}")
    
    print("\nTest Metrics:")
    print("Regression Metrics:")
    print(f"R² Score: {metrics['test_metrics']['regression']['r2']:.4f}")
    print(f"Adjusted R²: {metrics['test_metrics']['regression']['adjusted_r2']:.4f}")
    print(f"Explained Variance: {metrics['test_metrics']['regression']['explained_variance']:.4f}")
    print(f"MAE: {metrics['test_metrics']['regression']['mae']:.4f}")
    print(f"RMSE: {metrics['test_metrics']['regression']['rmse']:.4f}")
    
    print("\nClassification Metrics (Threshold = 7.0):")
    print(f"Accuracy: {metrics['test_metrics']['classification']['accuracy']:.4f}")
    print(f"Precision: {metrics['test_metrics']['classification']['precision']:.4f}")
    print(f"Recall: {metrics['test_metrics']['classification']['recall']:.4f}")
    
    if 'cross_validation' in metrics:
        print("\nCross-Validation Results:")
        print(f"Mean R² Score: {metrics['cross_validation']['r2_mean']:.4f}")
        print(f"R² Score Std: {metrics['cross_validation']['r2_std']:.4f}")
    
    print("\nTop 5 Most Important Features:")
    sorted_features = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")

def main():
    print("Initializing ISP Customer Satisfaction Model...")
    isp_model = ISPCustomerSatisfaction()
    
    print("\nTraining model...")
    metrics = isp_model.train_model('isp_admin_data.csv')
    
    print("\nSaving model...")
    isp_model.save_model('isp_satisfaction_model.joblib')
    
    format_metrics(metrics)
    print("\nModel has been trained and saved successfully!")

if __name__ == '__main__':
    main() 