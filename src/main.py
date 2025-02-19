from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *

import matplotlib.pyplot as plt
import pandas as pd

class Data:

    def __init__(self, dataset_path: str = 'data/dataset.csv') -> None:
        """Dataset Class Constructor"""
        self.dataset: pd.DataFrame = dataset_path
        self.original_dataset: pd.DataFrame = self.dataset

    @property
    def original_dataset(self) -> pd.DataFrame:
        """Returns Original Dataset Property"""
        return self._original
    
    @original_dataset.setter
    def original_dataset(self, dataset: pd.DataFrame) -> None:
        """Stores The Original Dataset Pre Processing as Df"""
        self._original: pd.DataFrame = dataset

    @property
    def dataset(self) -> pd.DataFrame:
        """Returns Dataset In Current State"""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset_path: str) -> None:
        """Loads A Dataset From A File, Converts Into Simple Series"""
        self._dataset: pd.DataFrame = pd.read_csv(dataset_path)
        self._dataset['Date'] = pd.to_datetime(self._dataset['Date'])
        self._dataset: pd.DataFrame = self._dataset.set_index('Date')
        self._dataset: pd.Series = self._dataset['Mean']
        self._dataset: pd.Series = validate_series(self._dataset)

class AnomalyDetection:

    def __init__(self, dataset: pd.Series):
        self._dataset: pd.Series = dataset
    
    def threshold_detection(self):
        algorithm: ThresholdAD = ThresholdAD(low=-0.5, high=0.75)
        return algorithm.detect(self._dataset)
    
    def quantile_detection(self):
        algorithm: QuantileAD = QuantileAD(low=0.01, high=0.99)
        return algorithm.fit_detect(self._dataset)
    
    def iqr_detection(self):
        algorithm: InterQuartileRangeAD = InterQuartileRangeAD(c=1)
        return algorithm.fit_detect(self._dataset)
    
    def esd_detection(self):
        #only use if data is normal distribution
        algorithm: GeneralizedESDTestAD = GeneralizedESDTestAD(alpha=0.3)
        return algorithm.fit_detect(self._dataset)
    
    def persistance_detection(self, c: float = 40.0, side: str = 'positive', window: int = 10):
        algorithm: PersistAD = PersistAD(c=c, side=side, window=window)
        return algorithm.fit_detect(self._dataset)

    def volatility_detection(self, c: float = 6.0, side: str = 'positive', window: int = 30):
        algorithm: VolatilityShiftAD = VolatilityShiftAD(c=c, side=side, window=window)
        return algorithm.fit_detect(self._dataset)
    
    def plot(self, data: pd.Series):
        plot(self._dataset, data, anomaly_color='red', anomaly_tag='marker')
        plt.show()

def main() -> None:
    Algorithm: AnomalyDetection = AnomalyDetection(Data().dataset)
    Algorithm.plot(Algorithm.threshold_detection())

if __name__ == '__main__':
    main()
