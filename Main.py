def main():
    """
    Main execution function to train and evaluate the LSTM model
    """

    dataset = pd.read_csv('group_2_project_final.csv')
    time_series_data = generate_time_series_data(dataset)

    # Hyperparameters
    # input_size was incorrectly set to the length of the dataset
    # Instead, it should be the number of features in your dataset
    input_size = 2 # Number of features, Coal production, and objective
    hidden_size = 10  # LSTM hidden units
    output_size = 1  # Predicting a single value
    num_layers = 2  # LSTM layers
    learning_rate = 0.0005
    num_epochs = 25
    batch_size = 32
    sequence_length = 10
    dropout_rate = 0.3

    input_scaler = MinMaxScaler()
    input_scaler.fit(time_series_data[['Coal_Production_change(%)', 'Nasa_Temp_Anomaly','Aviation_passanger_growth(%)']])

    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(
        time_series_data,
        batch_size=batch_size,
        sequence_length=sequence_length
    )

    # Initialize model
    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout_rate
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs
    )

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Assuming 'sequence' should be a batch of sequences from your dataset:
    #  - Get a batch from the train_loader for example
    #  - Extract the features from that batch
    predict_dataset = TimeSeriesDataset(
        time_series_data,
        sequence_length=sequence_length,
        predict_steps=1
    )
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=batch_size,
        shuffle=False
        )

    all_predictions = []

    # Create a new scaler for the output
    output_scaler = MinMaxScaler(feature_range=(-1,1))
    # Fit the scaler on the target values from your training data
    output_scaler.fit(dataset[['Nasa_Temp_Anomaly']])

    # Assuming you want to predict for the next year after the last data point in your dataset
    last_sequence = predict_dataset[-1][0].unsqueeze(0)  # Get the last sequence from the dataset

    prediction = predict_with_model(model, last_sequence, scaler=output_scaler)

    print("Predictions for 'Nasa_Temp_Anomaly'", prediction.item())


main()
