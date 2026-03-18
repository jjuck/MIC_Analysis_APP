from __future__ import annotations

import pandas as pd

from config.models import LimitPolicy, ProductCatalog
from core.analyzer import AnalysisService
from core.models import AnalysisReport, ProductDetection
from core.parser import ProductDetector, UploadedCsvReader


class AnalysisApplicationService:
    def __init__(self, product_catalog: ProductCatalog, limit_policy: LimitPolicy):
        self._reader = UploadedCsvReader()
        self._detector = ProductDetector(product_catalog)
        self._analyzer = AnalysisService(limit_policy)
        self._product_catalog = product_catalog

    def read_dataframe(self, uploaded_file) -> pd.DataFrame:
        return self._reader.read(uploaded_file)

    def detect_product(self, df: pd.DataFrame) -> ProductDetection:
        return self._detector.detect(df)

    def analyze(self, df: pd.DataFrame, model_name: str, detection: ProductDetection) -> AnalysisReport:
        product_spec = self._product_catalog.get(model_name)
        return self._analyzer.analyze(df, product_spec, detection)
