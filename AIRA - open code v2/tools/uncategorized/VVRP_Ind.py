        ### Visual Volume Range Profile
        # Calculate the average price
        dataframe['Average_Price'] = (dataframe['high'] + dataframe['low']) / 2
        hi = dataframe['high'].max()
        lo = dataframe['low'].min()
        width = hi - lo

        # Create bins and labels
        num_bins = 50
        bin_width = width / num_bins
        bin_labels = range(1, num_bins + 1)

        # Assign each row to a price bin
        dataframe['Price Bins'] = pd.cut(dataframe['Average_Price'], bins=num_bins, labels=bin_labels)

        # Calculate the total volume
        total_volume = dataframe['volume'].sum()

        # Create new columns for the volume bins
        for bin_label in bin_labels:
            dataframe[f'Volume Bin {bin_label}'] = 0  # Initialize all volume bins to 0

        # Reset starting_point
        starting_point = lo + bin_width / 2

        # Allocate volume to the corresponding volume bin for each row
        for index, row in dataframe.iterrows():
            if pd.notnull(row['Price Bins']):
                volume_bin_label = f'Volume Bin {int(row["Price Bins"])}'
                dataframe.at[index, volume_bin_label] = row['volume']

        # Create new columns for the mid price of each bin
        for bin_label in bin_labels:
            mid_price = starting_point + bin_width * (bin_label - 0.5)
            dataframe[f'Mid Price Bin {bin_label}'] = mid_price


        # Print the DataFrame with the adjusted volume bin lengths
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        volume_bins_sums = dataframe.filter(like='Volume Bin').sum(axis=0)

        # Add the aggregated volume bins to the DataFrame
        dataframe['Total Volume Bins'] = volume_bins_sums

        # Calculate the percentage of total volume in each volume bin
        for bin_label in bin_labels:
            percentage_col = f'Percentage Bin {bin_label}'
            volume_col = f'Volume Bin {bin_label}'
            dataframe[percentage_col] = dataframe[volume_col] / total_volume * 100

        perc_bins_sums = dataframe.filter(like='Percentage Bin').sum(axis=0)

        # print(pair, total_volume, dataframe['Average_Price'].iloc[-1], hi, lo, bin_width, width)
        # print(dataframe[[f'Mid Price Bin {bin_label}' for bin_label in bin_labels]].iloc[-1])
        # print(perc_bins_sums)
        # print(volume_bins_sums)

        ### example how access each value of the volume_bins and perc_bins
        fifth_value = volume_bins_sums.iloc[4]
        # print(f'Volume Bin 5: {fifth_value}')