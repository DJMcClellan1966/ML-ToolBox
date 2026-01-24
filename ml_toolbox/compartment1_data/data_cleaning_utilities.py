"""
Data Cleaning Utilities
Extracted from data science repositories

Features:
- Data cleaning workflows
- Missing data handling
- Data tidying methods
- Real-world cleaning patterns
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not available. Some functions will be limited.")


class DataCleaningUtilities:
    """
    Data Cleaning Utilities
    
    Comprehensive data cleaning functions extracted from
    various data science repositories
    """
    
    def __init__(self):
        """Initialize data cleaning utilities"""
        self.cleaning_history: List[Dict[str, Any]] = []
    
    def clean_missing_values(self, data: Union[np.ndarray, pd.DataFrame], 
                           strategy: str = 'mean', columns: Optional[List] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Clean missing values using various strategies
        
        Args:
            data: Input data (numpy array or pandas DataFrame)
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop', 'forward_fill')
            columns: Specific columns to clean (for DataFrames)
            
        Returns:
            Cleaned data
        """
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return self._clean_missing_pandas(data, strategy, columns)
        else:
            return self._clean_missing_numpy(data, strategy)
    
    def _clean_missing_pandas(self, df: pd.DataFrame, strategy: str, columns: Optional[List]) -> pd.DataFrame:
        """Clean missing values in pandas DataFrame"""
        target_cols = columns if columns else df.columns
        
        if strategy == 'mean':
            df[target_cols] = df[target_cols].fillna(df[target_cols].mean())
        elif strategy == 'median':
            df[target_cols] = df[target_cols].fillna(df[target_cols].median())
        elif strategy == 'mode':
            df[target_cols] = df[target_cols].fillna(df[target_cols].mode().iloc[0])
        elif strategy == 'drop':
            df = df.dropna(subset=target_cols)
        elif strategy == 'forward_fill':
            df[target_cols] = df[target_cols].fillna(method='ffill')
        elif strategy == 'backward_fill':
            df[target_cols] = df[target_cols].fillna(method='bfill')
        
        return df
    
    def _clean_missing_numpy(self, data: np.ndarray, strategy: str) -> np.ndarray:
        """Clean missing values in numpy array"""
        data = np.asarray(data)
        
        if strategy == 'mean':
            mask = ~np.isnan(data)
            if mask.any():
                mean_val = np.nanmean(data)
                data[~mask] = mean_val
        elif strategy == 'median':
            mask = ~np.isnan(data)
            if mask.any():
                median_val = np.nanmedian(data)
                data[~mask] = median_val
        elif strategy == 'drop':
            # For 2D arrays, drop rows with any NaN
            if len(data.shape) == 2:
                mask = ~np.isnan(data).any(axis=1)
                data = data[mask]
            else:
                data = data[~np.isnan(data)]
        
        return data
    
    def remove_outliers(self, data: Union[np.ndarray, pd.DataFrame], 
                       method: str = 'iqr', threshold: float = 1.5) -> Union[np.ndarray, pd.DataFrame]:
        """
        Remove outliers from data
        
        Args:
            data: Input data
            method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            threshold: Threshold for outlier detection
            
        Returns:
            Data with outliers removed
        """
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return self._remove_outliers_pandas(data, method, threshold)
        else:
            return self._remove_outliers_numpy(data, method, threshold)
    
    def _remove_outliers_pandas(self, df: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
        """Remove outliers from pandas DataFrame"""
        if method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df >= lower_bound) & (df <= upper_bound)
            return df[mask.all(axis=1)]
        elif method == 'zscore':
            z_scores = np.abs((df - df.mean()) / df.std())
            return df[(z_scores < threshold).all(axis=1)]
        return df
    
    def _remove_outliers_numpy(self, data: np.ndarray, method: str, threshold: float) -> np.ndarray:
        """Remove outliers from numpy array"""
        data = np.asarray(data)
        
        if method == 'iqr':
            if len(data.shape) == 1:
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (data >= lower_bound) & (data <= upper_bound)
                return data[mask]
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            mask = z_scores < threshold
            return data[mask]
        
        return data
    
    def standardize_data(self, data: Union[np.ndarray, pd.DataFrame], 
                        method: str = 'standard') -> Union[np.ndarray, pd.DataFrame]:
        """
        Standardize data (normalize)
        
        Args:
            data: Input data
            method: Standardization method ('standard', 'minmax', 'robust')
            
        Returns:
            Standardized data
        """
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            if method == 'standard':
                return (data - data.mean()) / data.std()
            elif method == 'minmax':
                return (data - data.min()) / (data.max() - data.min())
            elif method == 'robust':
                return (data - data.median()) / (data.quantile(0.75) - data.quantile(0.25))
        else:
            data = np.asarray(data)
            if method == 'standard':
                return (data - np.mean(data)) / np.std(data)
            elif method == 'minmax':
                return (data - np.min(data)) / (np.max(data) - np.min(data))
        
        return data
    
    def tidy_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Tidy data according to tidy data principles
        
        - Each variable forms a column
        - Each observation forms a row
        - Each type of observational unit forms a table
        """
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            # Ensure proper column names
            data.columns = [str(col).strip().lower().replace(' ', '_') for col in data.columns]
            # Remove duplicate rows
            data = data.drop_duplicates()
            # Reset index
            data = data.reset_index(drop=True)
        else:
            # For numpy arrays, ensure 2D
            data = np.asarray(data)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            # Remove duplicate rows
            data = np.unique(data, axis=0)
        
        return data
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names in DataFrame
        
        - Remove special characters
        - Convert to lowercase
        - Replace spaces with underscores
        """
        if not PANDAS_AVAILABLE:
            return df
        
        df.columns = [str(col).strip().lower().replace(' ', '_').replace('-', '_') 
                      for col in df.columns]
        # Remove special characters
        import re
        df.columns = [re.sub(r'[^a-z0-9_]', '', col) for col in df.columns]
        return df
    
    def get_cleaning_summary(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get summary of data cleaning needs
        
        Args:
            data: Input data
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_rows': len(data) if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame) else data.shape[0],
            'total_columns': len(data.columns) if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame) else data.shape[1],
            'missing_values': 0,
            'duplicate_rows': 0,
            'outliers_detected': False
        }
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            summary['missing_values'] = data.isnull().sum().sum()
            summary['duplicate_rows'] = data.duplicated().sum()
        else:
            data = np.asarray(data)
            summary['missing_values'] = np.isnan(data).sum()
            # Check for duplicates
            summary['duplicate_rows'] = len(data) - len(np.unique(data, axis=0))
        
        return summary


def get_data_cleaning_utilities() -> DataCleaningUtilities:
    """Get data cleaning utilities instance"""
    return DataCleaningUtilities()
