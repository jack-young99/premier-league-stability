def extract_pl_ids(df, column_name):
    """
    Returns a list of unique IDs from the specified column.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column_name (str): Name of the column containing IDs.
        
    Returns:
        list: Unique IDs.
    """
    return df[column_name].dropna().unique().tolist()